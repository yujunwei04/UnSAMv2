import logging

import numpy as np
import torch
import torch.distributed
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

from sam2.utils.misc import concat_points

from training.utils.data_utils import BatchedVideoDatapoint
import random

import sys
sys.path.append('/home/yujunwei/sam2/GraCo')
from isegm.inference.clicker import Clicker
# from training.utils.GraCo.isegm.inference.evaluation import get_sam_input
import cv2

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

def _iter_correct_pt_sampling_graco(
    self,
    is_init_cond_frame,
    point_inputs,
    gt_masks,
    high_res_features,
    pix_feat_with_mem,
    low_res_multimasks,
    high_res_multimasks,
    ious,
    low_res_masks,
    high_res_masks,
    object_score_logits,
    current_out,
):

    assert gt_masks is not None
        

    all_pred_masks = [low_res_masks]
    all_pred_high_res_masks = [high_res_masks]
    all_pred_multimasks = [low_res_multimasks]
    all_pred_high_res_multimasks = [high_res_multimasks]
    all_pred_ious = [ious]
    all_point_inputs = [point_inputs]
    all_object_score_logits = [object_score_logits]

    clicker_list = [Clicker(gt_mask=gt_mask) for gt_mask in gt_masks]
    pred_masks_list = [np.zeros_like(gt_mask) for gt_mask in gt_masks]
    point_coords = []
    points_labels = []

    for click_indx in range(self.num_correction_pt_per_frame):
        # sample a new point from the error between prediction and ground-truth
        # (with a small probability, directly sample from GT masks instead of errors)
        if self.training and self.prob_to_sample_from_gt_for_train > 0:
            sample_from_gt = (
                self.rng.random() < self.prob_to_sample_from_gt_for_train
            )
        else:
            sample_from_gt = False
        # if `pred_for_new_pt` is None, only GT masks will be used for point sampling
        pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)

        new_points, new_labels = get_next_point(
            gt_masks=gt_masks,
            pred_masks=pred_for_new_pt,
            method="uniform" if self.training else self.pt_sampling_for_eval,
        )

        point_inputs = concat_points(point_inputs, new_points, new_labels)
        # Feed the mask logits of the previous SAM outputs in the next SAM decoder step.
        # For tracking, this means that when the user adds a correction click, we also feed
        # the tracking output mask logits along with the click as input to the SAM decoder.
        mask_inputs = low_res_masks
        multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
        if self.use_act_ckpt_iterative_pt_sampling and not multimask_output:
            sam_outputs = torch.utils.checkpoint.checkpoint(
                self._forward_sam_heads,
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                use_reentrant=False,
            )
        else:
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            _,
            object_score_logits,
        ) = sam_outputs
        all_pred_masks.append(low_res_masks)
        all_pred_high_res_masks.append(high_res_masks)
        all_pred_multimasks.append(low_res_multimasks)
        all_pred_high_res_multimasks.append(high_res_multimasks)
        all_pred_ious.append(ious)
        all_point_inputs.append(point_inputs)
        all_object_score_logits.append(object_score_logits)

    # Concatenate the masks along channel (to compute losses on all of them,
    # using `MultiStepIteractiveMasks`)
    current_out["multistep_pred_masks"] = torch.cat(all_pred_masks, dim=1)
    current_out["multistep_pred_masks_high_res"] = torch.cat(
        all_pred_high_res_masks, dim=1
    )
    current_out["multistep_pred_multimasks"] = all_pred_multimasks
    current_out["multistep_pred_multimasks_high_res"] = all_pred_high_res_multimasks
    current_out["multistep_pred_ious"] = all_pred_ious
    current_out["multistep_point_inputs"] = all_point_inputs
    current_out["multistep_object_score_logits"] = all_object_score_logits

    return point_inputs, sam_outputs

def process_points(points):
    positive = points[:, :1, :]
    negative = points[:, 1:, :]

    filtered_points = []
    filtered_labels = []
    for batch in range(points.shape[0]):
        batch_points = []
        batch_labels = []
        for point in positive[batch]:
            if point[0] != -1:
                point_y, point_x = point[:2]
                batch_points.append([point_x, point_y])
                batch_labels.append(1)
        for point in negative[batch]:
            if point[0] != -1:
                point_y, point_x = point[:2]
                batch_points.append([point_x, point_y])
                batch_labels.append(0)
        filtered_points.append(np.array(batch_points))
        filtered_labels.append(np.array(batch_labels))

    return filtered_points, filtered_labels

def get_next_points_graco(pred, gt, points, click_indx, pred_thresh=0.49):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :]
    # pred = pred[:, 0, :, :]
    # gt = gt[:, 0, :, :]

    # fn_mask = np.logical_and(gt, pred < pred_thresh)
    # fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.logical_and(gt, np.logical_not(pred))
    fp_mask = np.logical_and(np.logical_not(gt), pred)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                # points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                # points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)
    return points

def graco_sample_optimized(gt_masks, pred_masks, mode, click_indx, points):
    """优化版本的graco_sample函数，减少torch-numpy转换"""
    device = points.device
    
    if mode == "train":
        # 完全在GPU上实现get_next_points_graco功能
        points = get_next_points_graco_torch(pred_masks, gt_masks, points, click_indx + 1)
        
        # 处理点数据，无需CPU转换
        filtered_points_list, filtered_labels_list = process_points_torch(points)
        
        # 将列表转换为批次的张量
        # 注：这里假设所有批次的点数量相同，如果不同需要padding
        new_points_graco = torch.stack(filtered_points_list, dim=0)
        new_labels_graco = torch.stack(filtered_labels_list, dim=0)
        
        return new_points_graco, new_labels_graco
    else:
        # 评估模式处理保持不变
        gt_masks_np = [gt_mask.cpu().numpy() for gt_mask in gt_masks]
        clicker_list = [Clicker(gt_mask=gt_mask_np) for gt_mask_np in gt_masks_np]
        
        point_coords = []
        points_labels = []
        
        for idx, gt_mask_np in enumerate(gt_masks_np):
            clicker = clicker_list[idx]
            pred_mask = pred_masks[idx].cpu().numpy()
            clicker.make_next_click(pred_mask)
            curr_point_coords, curr_point_labels = get_sam_input(clicker)
            point_coords.append(curr_point_coords)
            points_labels.append(curr_point_labels)
            
        # 一次性转换最终结果
        new_points_graco = torch.tensor(np.stack(point_coords, axis=0), device=device)
        new_labels_graco = torch.tensor(np.stack(points_labels, axis=0), device=device)
        return new_points_graco, new_labels_graco
    
def get_next_points_graco_torch(pred, gt, points, click_indx, pred_thresh=0.49):
    """PyTorch版本的get_next_points_graco，避免CPU-GPU传输"""
    assert click_indx > 0
    
    # 在GPU上进行所有操作
    pred = pred[:, 0, :, :]  # 不需要移动到CPU
    gt = gt[:, 0, :, :]
    
    # 使用PyTorch操作替代NumPy操作
    fn_mask = torch.logical_and(gt, torch.logical_not(pred))
    fp_mask = torch.logical_and(torch.logical_not(gt), pred)
    
    # 克隆点，保持在GPU上
    points = points.clone()
    num_points = points.size(1) // 2
    
    # 处理每个批次项
    for bindx in range(fn_mask.shape[0]):
        # 这部分需要转到CPU计算距离变换，然后马上返回GPU
        # 这是唯一需要CPU的部分
        fn_mask_cpu = fn_mask[bindx].cpu().numpy().astype(np.uint8)
        fp_mask_cpu = fp_mask[bindx].cpu().numpy().astype(np.uint8)
        
        # 填充和距离变换
        fn_mask_cpu = np.pad(fn_mask_cpu, ((1, 1), (1, 1)), 'constant')
        fp_mask_cpu = np.pad(fp_mask_cpu, ((1, 1), (1, 1)), 'constant')
        
        fn_mask_dt = cv2.distanceTransform(fn_mask_cpu, cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask_cpu, cv2.DIST_L2, 5)[1:-1, 1:-1]
        
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        
        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            # 随机选择一个点
            coords = indices[np.random.randint(0, len(indices))]
            # 立即将结果应用到GPU张量
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
    
    return points

def process_points_torch(points):
    """PyTorch版本的process_points，避免CPU-GPU传输"""
    positive = points[:, :1, :]
    negative = points[:, 1:, :]
    
    # 预分配结果列表
    batch_size = points.shape[0]
    filtered_points = []
    filtered_labels = []
    
    for batch in range(batch_size):
        # 处理正点
        pos_mask = positive[batch, :, 0] != -1
        pos_points = positive[batch, pos_mask, :2]
        
        # 处理负点
        neg_mask = negative[batch, :, 0] != -1
        neg_points = negative[batch, neg_mask, :2]
        
        # 交换x和y坐标
        if pos_points.size(0) > 0:
            pos_points = torch.stack([pos_points[:, 1], pos_points[:, 0]], dim=1)
            
        if neg_points.size(0) > 0:
            neg_points = torch.stack([neg_points[:, 1], neg_points[:, 0]], dim=1)
        
        # 创建标签张量
        pos_labels = torch.ones(pos_points.size(0), device=points.device, dtype=torch.int32)
        neg_labels = torch.zeros(neg_points.size(0), device=points.device, dtype=torch.int32)
        
        # 合并点和标签
        batch_points = torch.cat([pos_points, neg_points], dim=0) if pos_points.size(0) > 0 and neg_points.size(0) > 0 else \
                     pos_points if pos_points.size(0) > 0 else neg_points
        
        batch_labels = torch.cat([pos_labels, neg_labels], dim=0) if pos_labels.size(0) > 0 and neg_labels.size(0) > 0 else \
                     pos_labels if pos_labels.size(0) > 0 else neg_labels
        
        filtered_points.append(batch_points)
        filtered_labels.append(batch_labels)
    
    return filtered_points, filtered_labels


def graco_sample(gt_masks, pred_masks, mode, click_indx, points):
    # 为Clicker创建NumPy版本，但不修改原始gt_masks
    device = points.device
    gt_masks_np = [gt_mask.cpu().numpy() if torch.is_tensor(gt_mask) else gt_mask for gt_mask in gt_masks]
    # gt_masks_np = np.stack(gt_masks_np, axis=0)
    
    # 使用NumPy版本创建Clicker
    clicker_list = [Clicker(gt_mask=gt_mask_np) for gt_mask_np in gt_masks_np]
    
    # 对应的预测掩码也使用NumPy
    # pred_masks_np = [pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask for pred_mask in pred_masks]
    # pred_masks_np = np.stack(pred_masks_np, axis=0)
    
    point_coords = []
    points_labels = []
    
    ## GraCo's sampling method
    if mode == "train":
        # 这里使用原始张量, TODO: need to change to raw logits as inputs
        # prev_output = torch.sigmoid(pred_masks)
        # 确保points变量已定义
        # we should import points from sam2's initial sampling
        # points = torch.zeros(gt_masks.shape[0], 2 * 20, 3, device=gt_masks.device)  # 假设最多20个点
        points = get_next_points_graco(pred_masks, gt_masks, points, click_indx + 1)
        input_point, input_label = process_points(points.cpu().numpy())
        new_points_graco = np.stack(input_point, axis=0)
        new_labels_graco = np.stack(input_label, axis=0)
        # 转回张量
        return torch.from_numpy(new_points_graco).to(device), torch.from_numpy(new_labels_graco).to(device)
    else:
        for idx, gt_mask_np in enumerate(gt_masks_np):
            clicker = clicker_list[idx]
            pred_mask = pred_masks[idx,:,:].cpu().numpy()
            clicker.make_next_click(pred_mask)
            curr_point_coords, curr_point_labels = get_sam_input(clicker)
            point_coords.append(curr_point_coords)
            points_labels.append(curr_point_labels)
        new_points_graco = np.stack(point_coords, axis=0)
        new_labels_graco = np.stack(points_labels, axis=0)
        # 转回张量
        return torch.from_numpy(new_points_graco).to(device), torch.from_numpy(new_labels_graco).to(device)

# def graco_sample(gt_masks, pred_masks, mode, click_indx):
#     clicker_list = [Clicker(gt_mask=gt_mask.cpu()) for gt_mask in gt_masks]
#     pred_masks_list = [np.zeros_like(gt_mask) for gt_mask in gt_masks]
#     point_coords = []
#     points_labels = []
#     ## GraCo's sampling method
#     if mode == "train":
#         prev_output = torch.sigmoid(pred_masks)
#         points = get_next_points_graco(prev_output, gt_masks, points, click_indx + 1)
#         input_point, input_label = process_points(points.cpu().numpy())
#         new_points_graco = np.stack(input_point, axis=0)
#         new_labels_graco = np.stack(input_label, axis=0)
#         return new_points_graco.unsqueeze(1), new_labels_graco.unsqueeze(1)
#     else:
#         for idx, gt_mask in enumerate(gt_masks):
#             clicker = clicker_list[idx]
#             pred_mask = pred_masks_list[idx]
#             clicker.make_next_click(pred_mask)
#             curr_point_coords, curr_point_labels = get_sam_input(clicker)
#             point_coords.append(curr_point_coords)
#             points_labels.append(curr_point_labels)
#         new_points_graco = np.stack(point_coords, axis=0)
#         new_labels_graco = np.stack(points_labels, axis=0)
#         return new_points_graco.unsqueeze(1), new_labels_graco.unsqueeze(1)