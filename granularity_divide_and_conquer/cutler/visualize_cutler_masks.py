import os
import sys
import argparse
from typing import List, Tuple

import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

# 触发自定义 ROIHeads 注册
import cutler.custom_heads  # noqa: F401


def add_cutler_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = 50

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    cfg.TEST.NO_SEGM = False


def build_cutler_cfg(config_file: str, confidence_threshold: float, opts: List[str], weights: str = None):
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts or [])
    if weights is not None and len(weights) > 0:
        cfg.MODEL.WEIGHTS = weights
    # CPU 兼容
    if cfg.MODEL.DEVICE == 'cpu' and getattr(cfg.MODEL.RESNETS, 'NORM', None) == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = 'BN'
        if hasattr(cfg.MODEL, 'FPN') and hasattr(cfg.MODEL.FPN, 'NORM'):
            cfg.MODEL.FPN.NORM = 'BN'
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def is_contained(mask_a: np.ndarray, mask_b: np.ndarray, threshold: float = 0.8) -> bool:
    inter = np.count_nonzero(mask_a & mask_b)
    a = np.count_nonzero(mask_a)
    if a == 0:
        return False
    return (inter / a) > threshold


def nms_by_iou(masks: List[np.ndarray], threshold: float, step: int) -> List[np.ndarray]:
    if not masks:
        return []
    sorted_masks = sorted(masks, key=lambda m: area(m), reverse=True)
    keep = list(range(len(sorted_masks)))
    def iou(m1, m2):
        inter = np.count_nonzero(m1 & m2)
        u = np.count_nonzero(m1) + np.count_nonzero(m2) - inter
        return 0.0 if u == 0 else inter / u
    for i in range(len(sorted_masks)):
        if i in keep:
            for j in range(i + 1, min(len(sorted_masks), i + step)):
                if j in keep and iou(sorted_masks[i], sorted_masks[j]) > threshold:
                    keep.remove(j)
    return [sorted_masks[i] for i in keep]


def overlay(image_bgr: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    color = (255, 0, 0)
    out = image_bgr.copy()
    out[mask_bool] = (out[mask_bool] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
    return out


def find_instances_and_parts(all_masks: List[np.ndarray], min_instance_area: float = 0.05,
                             contain_thresh: float = 0.8) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    instance_masks: List[np.ndarray] = []
    instance_to_parts: List[List[np.ndarray]] = []

    for i, m in enumerate(all_masks):
        if area(m) <= min_instance_area:
            continue
        inst = True
        for j, n in enumerate(all_masks):
            if i != j and is_contained(m, n, threshold=contain_thresh):
                inst = False
                break
        if inst:
            instance_masks.append(m)
            instance_to_parts.append([])

    for idx, inst_m in enumerate(instance_masks):
        parts = []
        for m in all_masks:
            if m is inst_m:
                continue
            if is_contained(m, inst_m, threshold=contain_thresh):
                parts.append(m)
        instance_to_parts[idx] = parts

    return instance_masks, instance_to_parts


def save_masks_for_image(image_bgr: np.ndarray, image_name: str,
                         instances: List[np.ndarray], instance_to_parts: List[List[np.ndarray]],
                         out_dir: str, save_mode: str = 'divide') -> int:
    base = os.path.splitext(os.path.basename(image_name))[0]
    save_root = os.path.join(out_dir, base)
    os.makedirs(save_root, exist_ok=True)
    count = 0

    if save_mode in ('divide', 'instance', 'all'):
        for inst_idx, m in enumerate(instances):
            vis = overlay(image_bgr, m.astype(bool))
            cv2.imwrite(os.path.join(save_root, f"{base}_whole_inst{inst_idx:03d}.png"), vis)
            count += 1

    if save_mode in ('divide', 'part', 'all'):
        for inst_idx, parts in enumerate(instance_to_parts):
            for part_idx, m in enumerate(parts):
                vis = overlay(image_bgr, m.astype(bool))
                cv2.imwrite(os.path.join(save_root, f"{base}_part_inst{inst_idx:03d}_part{part_idx:03d}.png"), vis)
                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description='端到端：使用 CutLER divide 并逐个保存掩码可视化')
    parser.add_argument('--config-file', type=str,
                        default='model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml')
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--confidence-threshold', type=float, default=0.1)
    parser.add_argument('--NMS-iou', type=float, default=0.9)
    parser.add_argument('--NMS-step', type=int, default=5)
    parser.add_argument('--min-instance-area', type=float, default=0.05)
    parser.add_argument('--contain-thresh', type=float, default=0.8)
    parser.add_argument('--start-id', type=int, default=None)
    parser.add_argument('--end-id', type=int, default=None)
    parser.add_argument('--weights', type=str, default='', help='MODEL.WEIGHTS 路径（优先于 --opts）')
    parser.add_argument('--save-mode', type=str, default='divide', choices=['divide', 'instance', 'part', 'all'])
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    cfg = build_cutler_cfg(args.config_file, args.confidence_threshold, args.opts, args.weights)
    predictor = DefaultPredictor(cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if args.start_id is None:
        args.start_id = 0
    if args.end_id is None:
        args.end_id = len(files)

    total_saved = 0
    for idx, fname in enumerate(files, start=1):
        if idx - 1 < args.start_id:
            continue
        if idx - 1 >= args.end_id:
            break

        img_path = os.path.join(args.input_dir, fname)
        image = cv2.imread(img_path)
        if image is None:
            print(f"读取失败，跳过: {img_path}")
            continue

        pred = predictor(image)
        masks_tensor = pred["instances"].get("pred_masks").to("cpu")
        scores_tensor = pred["instances"].get("scores").to("cpu")
        scores = scores_tensor.numpy() if hasattr(scores_tensor, 'numpy') else np.array(scores_tensor)

        all_masks: List[np.ndarray] = []
        for i in range(masks_tensor.shape[0]):
            if float(scores[i]) > args.confidence_threshold:
                m = masks_tensor[i, :, :].numpy()
                m = (m > 0.5) if m.dtype != np.bool_ else m
                all_masks.append(m.astype(np.bool_))

        if not all_masks:
            print(f"阈值过滤后无掩码: {img_path}")
            continue

        instance_masks, instance_to_parts = find_instances_and_parts(all_masks,
                                                                     min_instance_area=args.min_instance_area,
                                                                     contain_thresh=args.contain_thresh)

        # 合并做 NMS（按面积降序）
        all_parts = [m for parts in instance_to_parts for m in parts]
        divide_masks = instance_masks + all_parts
        divide_masks = nms_by_iou(divide_masks, threshold=args.NMS_iou, step=args.NMS_step)

        saved = save_masks_for_image(image, fname, instance_masks, instance_to_parts,
                                     args.output_dir, save_mode=args.save_mode)
        total_saved += saved
        print(f"{fname}: 保存 {saved} 张掩码图像")

    print(f"完成，共保存 {total_saved} 张掩码可视化到 {args.output_dir}")


if __name__ == '__main__':
    main() 