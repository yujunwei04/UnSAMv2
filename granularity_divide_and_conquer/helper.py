"""Helpers for gra_divide_conquer.py -- granularity divide-and-conquer pipeline."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from coco_annotator import create_annotation_info

_TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(mask1) + np.count_nonzero(mask2) - intersection
    if union == 0:
        return 0.0
    return intersection / union


def coverage(mask1: np.ndarray, mask2: np.ndarray) -> float:
    denom = np.count_nonzero(mask1)
    if denom == 0:
        return 0.0
    return np.count_nonzero(np.logical_and(mask1, mask2)) / denom


def NMS(pool: Iterable[np.ndarray], threshold: float, step: int) -> List[np.ndarray]:
    sorted_masks = sorted(pool, key=area, reverse=True)
    masks_kept_indices = list(range(len(sorted_masks)))

    for i in range(len(sorted_masks)):
        if i in masks_kept_indices:
            for j in range(i + 1, min(len(sorted_masks), i + step)):
                if iou(sorted_masks[i], sorted_masks[j]) > threshold:
                    if j in masks_kept_indices:
                        masks_kept_indices.remove(j)

    return [sorted_masks[i] for i in masks_kept_indices]


def protected_NMS(
    protected_masks: List[np.ndarray],
    normal_masks: Iterable[np.ndarray],
    threshold: float,
    step: int,
) -> List[np.ndarray]:
    """NMS variant that keeps protected masks regardless of overlaps."""
    sorted_normal_masks = sorted(normal_masks, key=area, reverse=True)
    normal_kept_indices = list(range(len(sorted_normal_masks)))
    result_masks = list(protected_masks)

    # Remove normal masks that overlap too much with any protected mask.
    for i in range(len(sorted_normal_masks)):
        for p_mask in protected_masks:
            if iou(sorted_normal_masks[i], p_mask) > threshold:
                if i in normal_kept_indices:
                    normal_kept_indices.remove(i)
                break

    # Standard NMS within remaining normal masks.
    for i in range(len(sorted_normal_masks)):
        if i in normal_kept_indices:
            for j in range(i + 1, min(len(sorted_normal_masks), i + step)):
                if j in normal_kept_indices and iou(sorted_normal_masks[i], sorted_normal_masks[j]) > threshold:
                    normal_kept_indices.remove(j)

    result_masks.extend(sorted_normal_masks[i] for i in normal_kept_indices)
    return result_masks


def resize_mask(bipartition_masked: np.ndarray, target_size: Iterable[int]) -> np.ndarray:
    bipartition_masked_img = Image.fromarray(np.uint8(bipartition_masked * 255))
    resized = np.asarray(bipartition_masked_img.resize(target_size))
    resized = resized.astype(np.uint8)
    upper = np.max(resized)
    lower = np.min(resized)
    thresh = upper / 2.0
    resized[resized > thresh] = upper
    resized[resized <= thresh] = lower
    return resized


def smallest_square_containing_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if len(np.where(rows)[0]) == 0 or len(np.where(cols)[0]) == 0:
        return 0, 1, 0, 1

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return int(ymin), int(ymax), int(xmin), int(xmax)


def resize_with_aspect_ratio(image_array: np.ndarray, target_size: int, patch_size: int) -> Image.Image:
    image = Image.fromarray(image_array)
    original_width, original_height = image.size

    if original_width >= original_height:
        new_width = target_size
        new_height = int(original_height * (target_size / original_width))
    else:
        new_height = target_size
        new_width = int(original_width * (target_size / original_height))

    new_width = ((new_width + patch_size - 1) // patch_size) * patch_size
    new_height = ((new_height + patch_size - 1) // patch_size) * patch_size

    return image.resize([new_width, new_height])


def generate_dinov3_feature_matrix(
    backbone: torch.nn.Module,
    image: Image.Image,
    feat_dim: int,
    patch_size: int,
    visualize_similarity: bool | None = None,
) -> np.ndarray:
    """Generate DINOv3 feature grid for the provided image patch."""

    if next(backbone.parameters()).device == torch.device("cpu"):
        tensor = _TO_TENSOR(image).unsqueeze(0)
        feat_dict = backbone(tensor, is_training=True)
        feat = feat_dict["x_norm_patchtokens"][0].detach().cpu()
    else:
        tensor = _TO_TENSOR(image).unsqueeze(0).half().cuda()
        feat_dict = backbone(tensor, is_training=True)
        feat = feat_dict["x_norm_patchtokens"][0].detach().cpu()

    img_width, img_height = image.size
    feat_num_w = img_width // patch_size
    feat_num_h = img_height // patch_size

    feat_reshaped = feat.reshape(feat_num_h, feat_num_w, feat_dim)
    return feat_reshaped


def append_annotation(
    output: dict,
    segmentation_id: int,
    image_id: int,
    mask: np.ndarray,
    category_info: dict,
    *,
    is_divide: bool,
    instance_id: int,
    is_part: bool,
    granularity: float,
    is_conquer: bool,
) -> int:
    """Create, enrich, and append an annotation, returning the next id."""

    annotation_info = create_annotation_info(
        segmentation_id,
        image_id,
        category_info,
        mask.astype(np.uint8),
        None,
        is_divide=is_divide,
    )
    if annotation_info is None:
        return segmentation_id

    annotation_info["granularity"] = granularity
    annotation_info["instance_id"] = instance_id
    annotation_info["is_part"] = is_part
    annotation_info["is_conquer"] = is_conquer
    output["annotations"].append(annotation_info)
    return segmentation_id + 1
