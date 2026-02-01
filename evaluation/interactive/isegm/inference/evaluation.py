from time import time

import numpy as np
import torch
import cv2
from isegm.inference import utils
from isegm.inference.clicker import Clicker
import logging

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm
from time import time

import numpy as np
import torch
import torch.distributed as dist
import cv2
import os
from isegm.inference import utils
from isegm.inference.clicker import Clicker

import json
from pycocotools import mask as mask_util
import segmentation_refinement as refine

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

def evaluate_dataset_distributed(dataset, predictor, graco=False, sam_type=None, oracle=False, gra=None, phrase=None,
                    granularities=None, **kwargs):
    """Distributed dataset evaluation"""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        print(f"Using existing distributed environment: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        current_device = torch.cuda.current_device()
        print(f"Rank {rank}: Current CUDA device: {current_device}")
        
    elif 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Initializing distributed environment: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
    else:
        rank = 0
        world_size = 1
        print("Running in single GPU mode.")
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    gpu_indices = [i for i in range(rank, dataset_size, world_size)]
    
    all_ious = []
    start_time = time()
    
    if rank == 0:
        print(f"Number of samples total: {dataset_size}, processing {len(gpu_indices)} on rank {rank}")
        iterator = tqdm(gpu_indices, leave=False)
    else:
        iterator = gpu_indices
    
    try:
        for index in iterator:
            sample = dataset.get_sample(index)
            for object_id in sample.objects_ids:
                if graco:
                    sample_ious, gra_idx = evaluate_sam2_oracle(sample.image, sample.gt_mask(object_id), predictor, 
                                                               sample_id=index, oracle=oracle, refiner=None,
                                                               granularities=granularities, **kwargs)
                elif sam_type == 'SAM2':
                    _, sample_ious, _ = sam2_evaluate(sample.image, sample.gt_mask(object_id), predictor, 
                                                     sample_id=index, sam_type=sam_type, oracle=oracle, gra=gra,
                                                     phrase=phrase, **kwargs)
                else:
                    _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                       sample_id=index, sam_type=sam_type, oracle=oracle, gra=gra,
                                                       phrase=phrase, **kwargs)
                all_ious.append(sample_ious)
        
        end_time = time()
        elapsed_time = end_time - start_time
        
        if world_size > 1:
            print(f"Rank {rank} finished processing {len(gpu_indices)} samples, waiting for others...")
            dist.barrier()
        
        if world_size > 1:
            num_samples = len(all_ious)
            gathered_nums = [0] * world_size
            dist.all_gather_object(gathered_nums, num_samples)
            
            if rank == 0:
                print(f"Sample counts per rank: {gathered_nums}")
            
            all_ious_gathered = [None for _ in range(world_size)]
            elapsed_times = [None for _ in range(world_size)]
            
            dist.all_gather_object(all_ious_gathered, all_ious)
            dist.all_gather_object(elapsed_times, elapsed_time)
            
            if rank == 0:
                all_ious_merged = []
                for ious_list in all_ious_gathered:
                    if ious_list:
                        all_ious_merged.extend(ious_list)
                
                max_elapsed_time = max(elapsed_times) if elapsed_times else elapsed_time
                return all_ious_merged, max_elapsed_time
            else:
                return [], 0
        else:
            return all_ious, elapsed_time
            
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
        if world_size > 1:
            try:
                dist.barrier()
                if rank == 0:
                    return [], 0
                else:
                    return [], 0
            except:
                pass
        raise e

def evaluate_dataset(dataset, predictor, graco=False, sam_type=None, oracle=False, gra=None, phrase=None,
                    distributed=True, granularities=None, **kwargs):
    """Entry function for dataset evaluation"""
    if distributed:
        return evaluate_dataset_distributed(dataset, predictor, graco, sam_type, oracle, gra, phrase,
                                           granularities=granularities, **kwargs)
    
    all_ious = []
    start_time = time()
    print("Number of sample: ", len(dataset))
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)
        for object_id in sample.objects_ids:
            if graco:
                sample_ious, gra_idx = evaluate_sam2_oracle(sample.image, sample.gt_mask(object_id), predictor, 
                                                            sample_id=index, oracle=oracle,
                                                            granularities=granularities, **kwargs)
            elif sam_type == 'SAM2':
                _, sample_ious, _ = sam2_evaluate(sample.image, sample.gt_mask(object_id), predictor, 
                                                     sample_id=index, sam_type=sam_type, oracle=oracle, gra=gra,
                                                     phrase=phrase, **kwargs)
            else:
                _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask(object_id), predictor,
                                                    sample_id=index, sam_type=sam_type, oracle=oracle, gra=gra,
                                                    phrase=phrase, **kwargs)
            all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time
    return all_ious, elapsed_time

def sam2_evaluate(image, gt_mask, predictor, max_iou_thr, pred_thr=0.49, min_clicks=1, max_clicks=20, 
                  sample_id=None, sam_type=False, oracle=False, gra=None, phrase=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    print(f"sam2_evaluate with gra = {gra}")
    with torch.no_grad():
        predictor.set_image(image)
        if sam_type == 'SAM2':
            prev_low_res_masks_np = None
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                point_coords, point_labels = get_sam_input(clicker)
                if oracle:
                    ious = []
                    pred_masks = []
                    pred_probs, iou_predictions_np, low_res_masks_np = predictor.predict(point_coords, point_labels, 
                                                                                            mask_input=prev_low_res_masks_np,
                                                                                            multimask_output=True,
                                                                                            return_logits=True,
                                                                                            gra=None,
                                                                                            granularity=None,
                                                                                        )
                    for idx in range(pred_probs.shape[0]):
                        pred_masks.append(pred_probs[idx] > 0)
                        ious.append(utils.get_iou(gt_mask, pred_masks[-1]))
                    tgt_idx = np.argmax(np.array(ious))
                    best_mask = pred_masks[tgt_idx]
                    pred_mask = best_mask
                    iou = utils.get_iou(gt_mask, pred_mask)
                    prev_low_res_masks_np = np.expand_dims(low_res_masks_np[tgt_idx,:,:], axis=0)
                else:
                    pred_probs, iou_predictions_np, low_res_masks_np = predictor.predict(point_coords, point_labels, 
                                                                                            mask_input=prev_low_res_masks_np,
                                                                                            multimask_output=False,
                                                                                            return_logits=True,
                                                                                            gra=None,
                                                                                            granularity=None,
                                                                                        )
                    prev_low_res_masks_np = low_res_masks_np
                    pred_probs = pred_probs[0]
                    pred_mask = pred_probs > -1
                    iou = utils.get_iou(gt_mask, pred_mask)

                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    break
            return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

def evaluate_sam2_oracle(image, gt_mask, predictor, max_iou_thr,
                           pred_thr=0.49, min_clicks=1, max_clicks=20,
                           sample_id=None, callback=None, oracle=False, refiner=None, use_m2m=True,
                           granularities=None):
    clicker = Clicker(gt_mask=gt_mask)
    ious_lists = []
    click_indxs = []
    print("evaluate_sam2_oracle")
    with torch.no_grad():
        predictor.set_image(image)
        min_num = 100
        if granularities is None:
            gra_list = [round(gra * 0.1, 1) for gra in range(1, 11)]
        else:
            gra_list = granularities
        
        for cur_gra in gra_list:
            prev_low_res_masks_np = None
            ious_list = []
            clicker.reset_clicks()
            pred_mask = np.zeros_like(gt_mask)
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                point_coords, point_labels = get_sam_input(clicker)
                if oracle:
                    ious = []
                    pred_masks = []
                    multimask = False
                    if click_indx == 0:
                        multimask = True
                    pred_probs, iou_predictions_np, low_res_masks_np = predictor.predict(point_coords, point_labels, multimask_output=multimask,
                                                         return_logits=True, mask_input=prev_low_res_masks_np, gra=cur_gra, granularity=torch.tensor([cur_gra]).reshape(1, 1, 1))
                    for idx in range(pred_probs.shape[0]):
                        pred_masks.append(pred_probs[idx] > -1)
                        ious.append(utils.get_iou(gt_mask, pred_masks[-1]))
                    tgt_idx = np.argmax(np.array(ious))
                    iou = ious[tgt_idx]
                    prev_low_res_masks_np = np.expand_dims(low_res_masks_np[tgt_idx,:,:], axis=0)
                    pred_mask = pred_masks[tgt_idx]

                else:
                    pred_probs, iou_predictions_np, low_res_masks_np = predictor.predict(point_coords, point_labels, multimask_output=False,
                                                         return_logits=True, mask_input=prev_low_res_masks_np, gra=cur_gra, granularity=torch.tensor([cur_gra]).reshape(1, 1, 1))
                    prev_low_res_masks_np = low_res_masks_np
                    pred_probs = pred_probs[0]
                    pred_mask = pred_probs > -1
                    if refiner is not None:
                        pred_mask = postprocess(refiner, pred_mask, image)
                    iou = utils.get_iou(gt_mask, pred_mask)
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list, gra=cur_gra)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    min_num = min(min_num, click_indx + 1)
                    break
                if min_num <= max_clicks and click_indx + 1 > min_num:
                    break
            ious_lists.append(np.array(ious_list, dtype=np.float32))
            click_indxs.append(click_indx)
        click_indxs = np.array(click_indxs)
        tgt_idxs = np.squeeze(np.argwhere(click_indxs == np.min(click_indxs)), axis=1)
        selected_ious = [ious_lists[i] for i in tgt_idxs]
        max_index = np.argmax([ious[0] for ious in selected_ious])
        ious = selected_ious[max_index]
        tgt_idx = tgt_idxs[max_index]
    return ious, tgt_idx

def evaluate_sample_oracle(image, gt_mask, predictor, max_iou_thr,
                           pred_thr=0.49, min_clicks=1, max_clicks=20,
                           sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    ious_lists = []
    click_indxs = []
    with torch.no_grad():
        predictor.set_input_image(image)
        min_num = 100
        for gra in range(1, 11):
            cur_gra = round(gra * 0.1, 1)
            ious_list = []
            clicker.reset_clicks()
            pred_mask = np.zeros_like(gt_mask)
            predictor.prev_prediction = torch.zeros_like(predictor.original_image[:, :1, :, :])
            for click_indx in range(max_clicks):
                clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(clicker, gra=cur_gra)

                pred_mask = pred_probs > pred_thr
                iou = utils.get_iou(gt_mask, pred_mask)
                if callback is not None:
                    callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

                ious_list.append(iou)
                if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                    min_num = min(min_num, click_indx + 1)
                    break
                if min_num <= max_clicks and click_indx + 1 > min_num:
                    break
            ious_lists.append(np.array(ious_list, dtype=np.float32))
            click_indxs.append(click_indx)
        click_indxs = np.array(click_indxs)
        tgt_idxs = np.squeeze(np.argwhere(click_indxs == np.min(click_indxs)), axis=1)
        selected_ious = [ious_lists[i] for i in tgt_idxs]
        max_index = np.argmax([ious[0] for ious in selected_ious])
        ious = selected_ious[max_index]
        tgt_idx = tgt_idxs[max_index]
    return ious, tgt_idx

def get_sam_input(clicker, reverse=True):
    clicks_list = clicker.get_clicks()
    points_nd = get_points_nd([clicks_list])
    point_length = len(points_nd[0]) // 2
    point_coords = []
    point_labels = []
    for i, point in enumerate(points_nd[0]):
        if point[0] == -1:
            continue
        if i < point_length:
            point_labels.append(1)
        else:
            point_labels.append(0)
        if reverse:
            point_coords.append([point[1], point[0]])  # for SAM
    return np.array(point_coords), np.array(point_labels)

def get_points_nd(clicks_lists):
    total_clicks = []
    num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
    num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
    num_max_points = max(num_pos_clicks + num_neg_clicks)
    num_max_points = max(1, num_max_points)

    for clicks_list in clicks_lists:
        pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
        pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

        neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
        neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
        total_clicks.append(pos_clicks + neg_clicks)

    return total_clicks

def area(mask):
    return np.count_nonzero(mask) / mask.size

def iou(mask1, mask2):
    intersection = np.count_nonzero(np.logical_and(mask1, mask2))
    union = np.count_nonzero(mask1) + np.count_nonzero(mask2) - intersection
    if union == 0: return 0
    return intersection / union