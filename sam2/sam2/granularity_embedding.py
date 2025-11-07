import torch
import torch.nn as nn
import logging

class FourierGranularityMLP(nn.Module):
    def __init__(self, fourier_dim=128, decoder_dim=256, hidden_dim=None, 
                 num_layers=2, dropout=0.1, temperature=100):
        super().__init__()
        
        self.fourier_embedder = FourierEmbedder(hidden_dim=fourier_dim, temperature=temperature)
        
        self.mlp = GranularityMLP(
            granularity_dim=fourier_dim,
            decoder_dim=decoder_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
    def forward(self, granularity):
        if granularity.dim() == 0:
            granularity = granularity.view(1)
            
        fourier_features = self.fourier_embedder(granularity)
        
        return self.mlp(fourier_features)

class FourierEmbedder():
    def __init__(self, hidden_dim=128, temperature=100):
        self.hidden_dim = hidden_dim
        self.num_freqs = hidden_dim // 2 
        self.remaining_dim = hidden_dim % 2
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(self.num_freqs) / self.num_freqs)
        
    @torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        out = []
        # Add sin/cos pairs
        for freq in self.freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))
            
        if self.remaining_dim:
            out.append(torch.sin(self.temperature * x))
            
        return torch.cat(out, cat_dim)

class GranularityMLP(nn.Module):
    def __init__(self, granularity_dim, decoder_dim=256, hidden_dim=None, num_layers=2, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (granularity_dim + decoder_dim) // 2
        
        layers = []
        input_dim = granularity_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, decoder_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)