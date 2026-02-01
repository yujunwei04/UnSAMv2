#!/usr/bin/env python3
"""Evaluate whole-image segmentation using COCO-style metrics."""

import argparse
import json
import os
from tqdm import tqdm
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_util


def merge_dt_list(dt_dir):
    """Merge detection JSON files from a directory into a single COCO format."""
    merged_data = {
        "info": {"year": 2023, "version": "1", "date_created": ""},
        "images": [],
        "annotations": [],
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": [{"id": 1, "name": "object", "supercategory": ""}],
    }

    annotation_id_counter = 1
    image_ids = []

    for ann_path in tqdm(os.listdir(dt_dir), desc="Merging detections"):
        ann = json.load(open(os.path.join(dt_dir, ann_path)))
        for data in ann["annotations"]:
            if data["image_id"] not in image_ids:
                image_ids.append(data["image_id"])
                merged_data["images"].append({
                    "id": data["image_id"],
                    "height": data["segmentation"]["size"][0],
                    "width": data["segmentation"]["size"][1],
                })

            ann_area = mask_util.area(data["segmentation"]).tolist()
            data["id"] = annotation_id_counter
            annotation_id_counter += 1
            data["category_id"] = 1
            data["iscrowd"] = 0
            data["score"] = data.get("predicted_iou", data.get("score", 1.0))
            data["area"] = ann_area
            merged_data["annotations"].append(data)

    output_path = "merged_dt.json"
    print(f"Saving merged detections to {output_path}")
    with open(output_path, "w") as f:
        json.dump(merged_data, f)
    print(f"Total annotations: {annotation_id_counter - 1}")
    return output_path


def print_ar_iou50(coco_eval):
    """Print Average Recall at IoU=0.50 threshold."""
    if not coco_eval.eval:
        return

    p = coco_eval.params
    recall = coco_eval.eval.get("recall")
    if recall is None:
        return

    fmt = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
    iou_thr = 0.5

    iou_idx = np.where(np.isclose(p.iouThrs, iou_thr))[0]
    if len(iou_idx) == 0:
        return
    iou_idx = iou_idx[0]

    combos = [
        ("all", p.maxDets[0]),
        ("all", p.maxDets[1]),
        ("all", p.maxDets[2]),
        ("small", p.maxDets[2]),
        ("medium", p.maxDets[2]),
        ("large", p.maxDets[2]),
    ]

    for area_label, max_det in combos:
        if area_label not in p.areaRngLbl or max_det not in p.maxDets:
            continue
        area_idx = p.areaRngLbl.index(area_label)
        max_idx = p.maxDets.index(max_det)
        stats = recall[iou_idx, :, area_idx, max_idx]
        valid = stats[stats > -1]
        mean_val = -1.0 if valid.size == 0 else float(np.mean(valid))
        print(fmt.format("Average Recall", "(AR)", f"{iou_thr:0.2f}", area_label, max_det, mean_val))


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation predictions")
    parser.add_argument("--predict-directory", type=str, required=True,
                        help="Directory containing prediction JSON files")
    parser.add_argument("--gt-file", type=str, required=True,
                        help="COCO format ground truth JSON file")
    parser.add_argument("--iou-type", type=str, default="segm", help="IoU type for evaluation")
    args = parser.parse_args()

    # Merge detection files
    dt_path = merge_dt_list(args.predict_directory)
    coco_dt = COCO(dt_path)

    # Load ground truth (COCO format)
    coco_gt = COCO(args.gt_file)

    # Ensure iscrowd is set
    for ann in coco_gt.dataset.get("annotations", []):
        ann.setdefault("iscrowd", 0)
    coco_gt.createIndex()

    # Run evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, args.iou_type)
    coco_eval.params.useCats = 0
    coco_eval.params.maxDets = [1, 100, 1000]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap = coco_eval.stats[:6]
    ar = coco_eval.stats[6:12]

    print_ar_iou50(coco_eval)

    # Print copy-paste friendly results
    mAP_str = " ".join(f"{v*100:.2f}" for v in ap[1:])
    mAR_str = " ".join(f"{v*100:.2f}" for v in ar[2:])

    print(f"\nmAP copy-paste: {mAP_str}")
    print(f"mAR copy-paste: {mAR_str}")
    print(f"All copy-paste: {mAR_str} {mAP_str}")
    print(f"Total masks: {len(coco_dt.dataset['annotations'])}")


if __name__ == "__main__":
    main()
