# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Tuple, Type, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP


class LoRALinear(nn.Module):
    """LoRA (Low-Rank Adaptation) linear layer implementation."""
    
    def __init__(
        self, 
        original_layer: nn.Linear, 
        rank: int = 4, 
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        device = original_layer.weight.device
        dtype = original_layer.weight.dtype
            
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features, device=device, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        self.scaling = self.alpha / self.rank
        
    def forward(self, x: Tensor) -> Tensor:
        result = self.original_layer(x)
        lora_result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        
        return result + lora_result * self.scaling


class LoRAConfig:    
    def __init__(
        self,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
        target_modules: List[str] = None
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["q_proj", "k_proj", "v_proj", "out_proj"]



class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        lora_config: LoRAConfig = None,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
          lora_config (LoRAConfig): LoRA configuration for parameter-efficient fine-tuning
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.lora_config = lora_config
        self.lora_enabled = False
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def setup_lora_after_loading(self):
        if self.lora_config is not None:
            print("Applying LoRA to UnSAMv2...")
            for n, p in self.named_parameters():
                p.requires_grad = False
            self.apply_lora()
            
            for name, param in self.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    
            print("Successfully applied LoRA to UnSAMv2.")
        else:
            print("No LoRA config provided; skipping LoRA setup.")

    def apply_lora(self):
        if self.lora_config is None:
            raise ValueError("LoRA config is not provided")
            
        self.lora_enabled = True
        
        for layer in self.layers:
            self._apply_lora_to_attention_block(layer)
            
        self._apply_lora_to_attention(self.final_attn_token_to_image)
        
    def _apply_lora_to_attention_block(self, block: 'TwoWayAttentionBlock'):
        self._apply_lora_to_attention(block.self_attn)
        self._apply_lora_to_attention(block.cross_attn_token_to_image)
        self._apply_lora_to_attention(block.cross_attn_image_to_token)
        
    def _apply_lora_to_attention(self, attention_module):
        for module_name in self.lora_config.target_modules:
            if hasattr(attention_module, module_name):
                original_layer = getattr(attention_module, module_name)
                if isinstance(original_layer, nn.Linear):
                    lora_layer = LoRALinear(
                        original_layer=original_layer,
                        rank=self.lora_config.rank,
                        alpha=self.lora_config.alpha,
                        dropout=self.lora_config.dropout
                    )
                    setattr(attention_module, module_name, lora_layer)
                    
    def disable_lora(self):
        if not self.lora_enabled:
            return
            
        for layer in self.layers:
            self._disable_lora_in_attention_block(layer)
            
        self._disable_lora_in_attention(self.final_attn_token_to_image)
        self.lora_enabled = False
        
    def _disable_lora_in_attention_block(self, block: 'TwoWayAttentionBlock'):
        self._disable_lora_in_attention(block.self_attn)
        self._disable_lora_in_attention(block.cross_attn_token_to_image)
        self._disable_lora_in_attention(block.cross_attn_image_to_token)
        
    def _disable_lora_in_attention(self, attention_module):
        for module_name in self.lora_config.target_modules:
            if hasattr(attention_module, module_name):
                current_layer = getattr(attention_module, module_name)
                if isinstance(current_layer, LoRALinear):
                    setattr(attention_module, module_name, current_layer.original_layer)
                    
    def get_lora_parameters(self):
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params.append(param)
        return lora_params
        
    def save_lora_weights(self, path: str):
        if not self.lora_enabled:
            raise ValueError("LoRA is not enabled")
            
        lora_state_dict = {}
        for name, param in self.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data
                
        torch.save({
            'lora_state_dict': lora_state_dict,
            'lora_config': {
                'rank': self.lora_config.rank,
                'alpha': self.lora_config.alpha,
                'dropout': self.lora_config.dropout,
                'target_modules': self.lora_config.target_modules
            }
        }, path)
        
    def load_lora_weights(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        lora_state_dict = checkpoint['lora_state_dict']
        
        missing_keys, unexpected_keys = self.load_state_dict(lora_state_dict, strict=False)

        print(f"LoRA weights loaded successfully. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        return missing_keys, unexpected_keys

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout
        self.last_attention_map = None

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        # Store attention map for visualization
        self.last_attention_map = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout and attention
        if dropout_p > 0:
            attn_weights = F.dropout(self.last_attention_map, p=dropout_p, training=self.training)
        else:
            attn_weights = self.last_attention_map
            
        out = torch.matmul(attn_weights, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        # whether to repeat q rope to match k length
        # this is needed for cross-attention to memories
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # [w, h] for stride 16 feats at 1024 resolution
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = (
            freqs_cis.to("cuda") if torch.cuda.is_available() else freqs_cis
        )
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, num_k_exclude_rope: int = 0
    ) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q,
            k[:, :, :num_k_rope],
            freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )

        dropout_p = self.dropout_p if self.training else 0.0
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        # Store attention map for visualization
        self.last_attention_map = F.softmax(attn_weights, dim=-1)
        
        # Apply dropout and attention
        if dropout_p > 0:
            attn_weights = F.dropout(self.last_attention_map, p=dropout_p, training=self.training)
        else:
            attn_weights = self.last_attention_map
            
        out = torch.matmul(attn_weights, v)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
