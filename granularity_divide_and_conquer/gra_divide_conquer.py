from collections import defaultdict
import os
import argparse
import torch
import tqdm
import numpy as np
import segmentation_refinement as refine
import json
from tqdm import tqdm
import cv2
import cutler.custom_heads

from coco_annotator import create_image_info, create_annotation_info, output, category_info
from iterative_merging import iterative_merge
from cascadepsp import postprocess
from detectron2.config import get_cfg

from detectron2.engine.defaults import DefaultPredictor
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.config import get_cfg
from detectron2.utils.colormap import random_color
from mask2former import add_maskformer2_config
from pycocotools import mask as mask_util
from compute_granularity import compute_granularity_scores_sqrt
from helper import (
    NMS,
    protected_NMS,
    area,
    coverage,
    resize_mask,
    smallest_square_containing_mask,
    resize_with_aspect_ratio,
    generate_dinov3_feature_matrix,
    append_annotation,
)
import warnings



warnings.filterwarnings("ignore")


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

def setup_cutler(args):
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.MODEL.DEVICE == 'cpu' and getattr(cfg.MODEL.RESNETS, 'NORM', None) == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        if hasattr(cfg.MODEL, 'FPN') and hasattr(cfg.MODEL.FPN, 'NORM'):
            cfg.MODEL.FPN.NORM = "BN"
    if hasattr(cfg.MODEL, 'MASK_FORMER') and hasattr(cfg.MODEL.MASK_FORMER, 'TEST'):
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_SCORE_THRESH = args.confidence_threshold
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    if hasattr(cfg.MODEL, 'PANOPTIC_FPN') and hasattr(cfg.MODEL.PANOPTIC_FPN, 'COMBINE'):
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.TEST.DETECTIONS_PER_IMAGE = args.detection_per_image
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config-file",
        default="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml",
        metavar="FILE",
    )
    # backbone args
    parser.add_argument("--patch-size", default=8, type=int)
    parser.add_argument("--feature-dim", default=768, type=int)
    parser.add_argument("--backbone-size", default='base', type=str)
    parser.add_argument("--backbone-url", default="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth", type=str)

    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--sa1b-anno-dir", type=str)
    parser.add_argument("--output-dir", type=str, default="output_dico")
    parser.add_argument("--preprocess", default=None, type=bool)
    parser.add_argument("--postprocess", default=None, type=bool)
    parser.add_argument("--granularity", default=None, type=bool)
    parser.add_argument("--detection-per-image", default=1000, type=int)
    # preprocess args
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--start-id", default=None, type=int)
    parser.add_argument("--end-id", default=None, type=int)
    parser.add_argument("--local-size", default=256, type=int)
    parser.add_argument("--kept-thresh", default=0.9)
    parser.add_argument("--NMS-iou", default=0.9, type=float)
    parser.add_argument("--NMS-step", default=5)
    parser.add_argument("--thetas", nargs='+', type=float, default=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    parser.add_argument("--NMS-iou-final", default=0.9, type=float)
    # postprocess args
    parser.add_argument("--crop-ratio", default=2.0)
    parser.add_argument("--refine-scale", default=1)
    parser.add_argument("--refine-min-L", default=100)
    parser.add_argument("--refine-max-L", default=900)
    parser.add_argument("--iou-thresh", default=0.5)
    parser.add_argument("--min-area-thresh", default=0.0)
    parser.add_argument("--max-area-thresh", default=0.9)
    parser.add_argument("--cover-thresh", default=0.9)
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use-norm",
        action="store_true",
        default=True
    )
    return parser

def main():
    args = get_parser().parse_args()
    print(args)
    
    if getattr(args, 'vis_dir', None) is None:
        args.vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(args.vis_dir, exist_ok=True)
    
    # Initialize refiner for postprocessing
    refiner = refine.Refiner(device='cuda:0')

    # divide-and-conquer algorithm
    if args.preprocess:
        if not args.start_id:
            args.start_id = 0
        if not args.end_id: 
            args.end_id = len(os.listdir(args.input_dir))
        if not os.path.exists(args.output_dir): 
            os.makedirs(args.output_dir)

        # load the CutLER model
        cfg = setup_cutler(args)
        predictor = DefaultPredictor(cfg)

        # load DINO backbone
        DINO_DIR = "./dinov3"
        dinov3 = torch.hub.load(DINO_DIR, 'dinov3_vitb16', source='local', weights='./dinov3/weights/dinov3_vitb16_pretrain.pth')

        segmentation_id = 1
        cnt = 0

        for image_name in tqdm(os.scandir(args.input_dir)):
            image_name = image_name.name
            if image_name == '.DS_Store':
                continue
            cnt += 1

            if args.start_id and args.end_id:
                if cnt < args.start_id: continue
                if cnt >= args.end_id: break
            # coco format annotator initialization
            output["image"], output["annotations"] = {}, []

            # save path initialization
            image_id = int(image_name.replace(".jpg", "").replace("sa_", ""))

            save_path = f"{args.output_dir}/{image_name.replace('.jpg', '.json')}"
            if os.path.exists(save_path):
                continue

            # Image import
            image_path = os.path.join(args.input_dir, image_name)
            image = cv2.imread(image_path)
            H, W = image.shape[:2]

            # Divide phase
            predictions = predictor(image)
            divide_masks_tensor = predictions["instances"].get("pred_masks")
            conf_scores = predictions["instances"].get("scores")
            all_unsam_masks = []

            for i in range(divide_masks_tensor.shape[0]):
                if conf_scores[i] > args.confidence_threshold:
                    all_unsam_masks.append(divide_masks_tensor[i,:,:].cpu().numpy())
            print(f"Total CutLER masks with confidence > {args.confidence_threshold}: {len(all_unsam_masks)}")

            instance_masks = []
            instance_to_parts = {}  # instance_idx -> [part_mask_indices]
            
            def is_contained(mask_a, mask_b, threshold=0.8):
                intersection = np.count_nonzero(np.logical_and(mask_a, mask_b))
                area_a = np.count_nonzero(mask_a)
                if area_a == 0:
                    return False
                return (intersection / area_a) > threshold
            
            for i, mask in enumerate(all_unsam_masks):
                mask_area = area(mask)
                if mask_area <= 0.05:
                    continue
                    
                is_instance = True
                for j, other_mask in enumerate(all_unsam_masks):
                    if i != j and is_contained(mask, other_mask):
                        is_instance = False
                        break
                
                if is_instance:
                    instance_idx = len(instance_masks)
                    instance_masks.append(mask)
                    instance_to_parts[instance_idx] = []
                        
            for inst_idx, inst_mask in enumerate(instance_masks):
                for i, mask in enumerate(all_unsam_masks):
                    if np.array_equal(mask, inst_mask):
                        continue
                    if is_contained(mask, inst_mask):
                        instance_to_parts[inst_idx].append(mask)
           
            divide_masks = instance_masks.copy()
            all_part_masks = []
            for parts in instance_to_parts.values():
                all_part_masks.extend(parts)
            divide_masks.extend(all_part_masks)
            
            divide_masks = NMS(divide_masks, args.NMS_iou, args.NMS_step)
            layer_cnt = defaultdict(int)
            divide_conquer_masks = []
            
            for inst_idx, inst_mask in enumerate(instance_masks):
                try:
                    pre_inst_temp_annotations = {"annotations": []}
                    pre_inst_sid = 1
                    pre_inst_ann = create_annotation_info(
                        pre_inst_sid, image_id, category_info, inst_mask.astype(np.uint8), None, is_divide=True
                    )
                    if pre_inst_ann is not None:
                        pre_inst_temp_annotations["annotations"].append(pre_inst_ann)
                        pre_refined_inst = postprocess(args, refiner, pre_inst_temp_annotations, image)
                        if isinstance(pre_refined_inst, dict) and len(pre_refined_inst.get("annotations", [])) > 0:
                            inst_mask = mask_util.decode(pre_refined_inst["annotations"][0]["segmentation"]).astype(bool)
                except Exception:
                    pass
                conquer_masks = []
                
                ymin, ymax, xmin, xmax = smallest_square_containing_mask(inst_mask)
                if (ymax-ymin) <= 0 or (xmax-xmin) <= 0:
                    continue
                local_image = image[ymin:ymax, xmin:xmax]
                
                resized_local_image = resize_with_aspect_ratio(local_image, args.local_size, 16)
                dinov3_feat_mat = generate_dinov3_feature_matrix(dinov3, resized_local_image, args.feature_dim, args.patch_size,
                                                                 visualize_similarity=False)
                merging_masks = iterative_merge(dinov3_feat_mat, args.thetas, min_size=4)
                
                for layer_idx, layer in enumerate(merging_masks):
                    if layer.shape[0] == 0: continue

                    for i in range(layer.shape[0]):
                        mask = layer[i, :, :]
                        mask = resize_mask(mask, [xmax-xmin, ymax-ymin])
                        mask = (mask > 0.5 * 255).astype(int)

                        if coverage(mask, inst_mask[ymin:ymax, xmin:xmax]) <= args.kept_thresh: continue
                        enlarged_mask = np.zeros_like(inst_mask)
                        enlarged_mask[ymin:ymax, xmin:xmax] = mask
                        conquer_masks.append(enlarged_mask)
                        layer_cnt[layer_idx] += 1

                conquer_masks = NMS(conquer_masks, args.NMS_iou, args.NMS_step)
                part_masks_for_this_instance = instance_to_parts[inst_idx]
                protected_masks = [inst_mask]
                normal_masks = conquer_masks.copy()
                normal_masks.extend(part_masks_for_this_instance)

                if len(normal_masks) > 0:
                    final_masks = protected_NMS(protected_masks, normal_masks, args.NMS_iou, args.NMS_step)
                else:
                    final_masks = protected_masks

                divide_granularity_score = None
                refined_annotations = {"annotations": []}
                current_instance_mask = final_masks[0] if len(final_masks) > 0 else inst_mask
                final_part_masks = final_masks[1:] if len(final_masks) > 1 else []
                                
                if len(final_part_masks) > 0:
                    temp_annotations = {"annotations": []}
                    temp_segmentation_id = 1

                    conquer_mask_count = len(conquer_masks)

                    for mask_idx, m in enumerate(final_part_masks):
                        annotation_info = create_annotation_info(
                            temp_segmentation_id, image_id, category_info, m.astype(np.uint8), None, is_divide=False)
                        if annotation_info is not None:
                            annotation_info['is_conquer'] = mask_idx < conquer_mask_count
                            temp_annotations["annotations"].append(annotation_info)
                            temp_segmentation_id += 1

                    has_temp_annotations = len(temp_annotations["annotations"]) > 0
                    processed_parts = False

                    if has_temp_annotations:
                        refined_annotations = postprocess(args, refiner, temp_annotations, image)
                        refined_list = refined_annotations.get("annotations", [])

                        if refined_list:
                            decoded_masks = [
                                mask_util.decode(annotation['segmentation']).astype(bool)
                                for annotation in refined_list
                            ]

                            if decoded_masks:
                                decoded_masks.append(current_instance_mask.astype(bool))
                                decoded_masks = np.stack(decoded_masks).astype(bool)
                                granularity_scores = compute_granularity_scores_sqrt(decoded_masks, mode="sqrt").tolist()

                                part_granularity_scores = granularity_scores[:-1]
                                divide_granularity_score = granularity_scores[-1]

                                for i, granularity_score in enumerate(part_granularity_scores):
                                    refined_list[i]['granularity'] = granularity_score
                                    refined_list[i]['instance_id'] = inst_idx
                                    refined_list[i]['is_part'] = True

                                for annotation in refined_list:
                                    mask = mask_util.decode(annotation['segmentation'])
                                    divide_conquer_masks.append(mask)
                                    segmentation_id = append_annotation(
                                        output,
                                        segmentation_id,
                                        image_id,
                                        mask,
                                        category_info,
                                        is_divide=False,
                                        instance_id=annotation['instance_id'],
                                        is_part=annotation['is_part'],
                                        granularity=annotation['granularity'],
                                        is_conquer=annotation.get('is_conquer', False),
                                    )
                                processed_parts = True

                    if not processed_parts:
                        fallback_masks = final_masks if has_temp_annotations else final_part_masks
                        for mask_idx, m in enumerate(fallback_masks):
                            segmentation_id = append_annotation(
                                output,
                                segmentation_id,
                                image_id,
                                m,
                                category_info,
                                is_divide=False,
                                instance_id=inst_idx,
                                is_part=True,
                                granularity=0.5,
                                is_conquer=mask_idx < len(conquer_masks),
                            )
                        divide_conquer_masks.extend(fallback_masks)

                if divide_granularity_score is None:
                    inst_mask_array = np.stack([current_instance_mask.astype(bool)])
                    divide_granularity_score = compute_granularity_scores_sqrt(inst_mask_array, mode="sqrt").tolist()[0]

                segmentation_id = append_annotation(
                    output,
                    segmentation_id,
                    image_id,
                    current_instance_mask,
                    category_info,
                    is_divide=True,
                    instance_id=inst_idx,
                    is_part=False,
                    granularity=divide_granularity_score,
                    is_conquer=False,
                )

            image_info = create_image_info(
                image_id, "{}".format(image_name), (H, W, 3))
            output["image"] = image_info
            print(f'We obtained all of our pseudo labels for image {image_id}. We have {len(divide_conquer_masks)} masks!')
            with open(save_path, 'w') as output_json_file:
                json.dump(output, output_json_file, indent=2)

if __name__ == "__main__":
    main()
