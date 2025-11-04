import torch
import numpy as np

def compute_granularity_scores_sqrt(masks, mode="sqrt"):
    num_masks, H, W = masks.shape
    granularity_scores = np.zeros(num_masks)
    
    areas = np.sum(masks.reshape(num_masks, -1), axis=1)
    sorted_indices = np.argsort(areas)
    sorted_areas = areas[sorted_indices]
    min_area = sorted_areas[0]
    max_area = sorted_areas[-1]
    
    if min_area == max_area:
        granularity_scores[:] = 1.0
        return granularity_scores
    
    sqrt_min = np.sqrt(min_area + 1e-10)
    sqrt_max = np.sqrt(max_area + 1e-10)
    sqrt_diff = sqrt_max - sqrt_min
    gra_diff = 0.9
    
    nan_cnt = 0
    
    for i, mask_idx in enumerate(sorted_indices):
        curr_area = areas[mask_idx]
        
        if i == 0:
            granularity_scores[mask_idx] = 0.1
        elif i == len(sorted_indices) - 1:
            granularity_scores[mask_idx] = 1.0
        else:
            sqrt_curr = np.sqrt(curr_area + 1e-10)
            gra = ((sqrt_curr - sqrt_min) / sqrt_diff) * gra_diff + 0.1
            
            if np.isnan(gra) or np.isinf(gra):
                granularity_scores[mask_idx] = 1.0
                nan_cnt += 1
            else:
                granularity_scores[mask_idx] = gra
    return granularity_scores
