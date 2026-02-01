#!/usr/bin/env python3

"""Generate per-image COCO-style annotations using SAM2 automatic masks."""

import argparse
import datetime
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from pycocotools import mask
import pycocotools.mask as mask_util

try:
    import segmentation_refinement as refine  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    refine = None

from cascadepsp import postprocess
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):
    """Return image_info in COCO style."""
    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[1],
        "height": image_size[0],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }
    return image_info


def create_annotation_info(
    annotation_id,
    image_id,
    category_info,
    binary_mask,
    image_size=None,
    bounding_box=None,
    is_divide=False,
):
    """Return annotation info in COCO style."""
    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[..., None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("ascii")
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
        "is_divide": is_divide,
    }

    return annotation_info


INFO = {
    "version": "1.0",
    "year": 2023,
    "date_created": datetime.datetime.utcnow().isoformat(" "),
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
    }
]

CATEGORIES = [
    {
        "id": 1,
        "name": "fg",
        "supercategory": "fg",
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]

category_info = {
    "is_crowd": 0,
    "id": 1,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_image_paths(directory: Path, recursive: bool) -> List[Path]:
    candidates: Iterable[Path]
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    images = sorted(
        [p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: natrual_key(path.name),
    )
    return images


def _load_image_id_map(path: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    with path.open("r", encoding="utf-8") as fh:
        raw_mapping = json.load(fh)

    mapping_by_name: Dict[str, int] = {}
    mapping_by_stem: Dict[str, int] = {}
    for raw_key, image_id in raw_mapping.items():
        key_path = Path(raw_key)
        name = key_path.name
        stem = key_path.stem
        mapping_by_name[name] = image_id
        mapping_by_stem[stem] = image_id

    logging.info(
        "Loaded %d entries from image id map %s", len(raw_mapping), path
    )
    return mapping_by_name, mapping_by_stem


def _parse_image_id(image_path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)", image_path.stem)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logging.debug("Unable to parse numeric id from %s", image_path.stem)
    return fallback


def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _build_mask_generator(
    model_cfg: str,
    checkpoint_path: str,
    device: str,
) -> SAM2AutomaticMaskGenerator:
    logging.info("Loading SAM2 model from %s", model_cfg)
    sam_model = build_sam2(model_cfg, checkpoint_path, device=device, mode="eval", strict=True)
    generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=64,
        points_per_batch=128,
        mask_threshold=-1,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.9,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        use_m2m=True,
    )
    return generator


def _build_refiner(device: str):
    if refine is None:
        logging.warning("segmentation_refinement is not installed; CascadePSP post-processing disabled")
        return None
    try:
        return refine.Refiner(device=device)
    except Exception as exc:  # pragma: no cover - depends on external package
        logging.warning("Failed to initialize CascadePSP refiner: %s", exc)
        return None


def _postprocess_masks(masks: List[dict], image: np.ndarray, refiner) -> List[dict]:
    if not masks or refiner is None:
        return masks

    annotations = {"annotations": []}
    for idx, mask_data in enumerate(masks):
        mask_array = mask_data["segmentation"].astype(np.uint8)
        encoded_mask = mask_util.encode(np.asfortranarray(mask_array))
        encoded_mask["counts"] = encoded_mask["counts"].decode("ascii")

        annotation = {
            "id": idx,
            "segmentation": encoded_mask,
            "bbox": [float(x) for x in mask_data["bbox"]],
            "area": float(mask_data["area"]),
            "predicted_iou": float(mask_data.get("predicted_iou", 0.0)),
            "stability_score": float(mask_data.get("stability_score", 0.0)),
        }
        annotations["annotations"].append(annotation)

    class Args:
        def __init__(self) -> None:
            self.crop_ratio = 2.0
            self.refine_scale = 1
            self.refine_min_L = 100
            self.refine_max_L = 900
            self.iou_thresh = 0.5
            self.min_area_thresh = 0.0
            self.max_area_thresh = 0.9
            self.cover_thresh = 0.9

    refined = postprocess(Args(), refiner, annotations, image)

    refined_masks: List[dict] = []
    for idx, annotation in enumerate(refined.get("annotations", [])):
        refined_mask = mask_util.decode(annotation["segmentation"])
        refined_masks.append(
            {
                "segmentation": refined_mask.astype(bool),
                "area": float(annotation.get("area", annotations["annotations"][idx]["area"])),
                "bbox": annotation.get("bbox", annotations["annotations"][idx]["bbox"]),
                "predicted_iou": float(annotation.get("predicted_iou", annotations["annotations"][idx]["predicted_iou"])),
                "stability_score": float(annotation.get("stability_score", annotations["annotations"][idx]["stability_score"])),
            }
        )
    return refined_masks


def _prepare_output_template() -> dict:
    info = dict(INFO)
    info["date_created"] = datetime.datetime.utcnow().isoformat(" ")
    return {
        "info": info,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": [],
    }


def _annotate_image(
    image_path: Path,
    image_id: int,
    mask_generator: SAM2AutomaticMaskGenerator,
    granularities: Sequence[float],
    use_postprocess: bool,
    refiner,
) -> dict:
    image = _load_image(image_path)
    output = _prepare_output_template()
    image_info = create_image_info(image_id, image_path.name, image.shape[:2])
    output["images"].append(image_info)

    segmentation_id = 1
    annotation_id_offset = image_id * 10000
    for granularity in granularities:
        logging.info(
            "Generating masks for %s with granularity %.2f",
            image_path.name,
            granularity,
        )
        if granularity < 0.5:
            mask_generator.pred_iou_thresh = 0.65
            mask_generator.stability_score_thresh = 0.7
        else:
            mask_generator.pred_iou_thresh = 0.5
            mask_generator.stability_score_thresh = 0.7
        logging.debug(
            "Using pred_iou_thresh=%.2f, stability_score_thresh=%.2f",
            mask_generator.pred_iou_thresh,
            mask_generator.stability_score_thresh,
        )
        masks = mask_generator.generate(image, gra=granularity)
        logging.info("Found %d masks", len(masks))

        if use_postprocess:
            masks = _postprocess_masks(masks, image, refiner)
            logging.info("Post-processed mask count: %d", len(masks))

        for mask_data in masks:
            mask_array = mask_data["segmentation"].astype(bool).astype(np.uint8)
            annotation_info = create_annotation_info(
                annotation_id_offset + segmentation_id,
                image_id,
                category_info,
                mask_array,
                None,
                is_divide=False,
            )
            if annotation_info is None:
                continue
            annotation_info["predicted_iou"] = float(mask_data.get("predicted_iou", 0.0))
            annotation_info["stability_score"] = float(mask_data.get("stability_score", 0.0))
            annotation_info["area"] = float(mask_data.get("area", annotation_info["area"]))
            annotation_info["bbox"] = [float(x) for x in mask_data.get("bbox", annotation_info["bbox"])]
            annotation_info["source_granularity"] = float(granularity)
            output["annotations"].append(annotation_info)
            segmentation_id += 1

    return output


def run_annotation(
    image_dir: Path,
    output_dir: Path,
    model_cfg: str,
    checkpoint_path: str,
    granularities: Sequence[float],
    recursive: bool,
    use_postprocess: bool,
    device: Optional[str] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    image_id_map_path: Optional[Path] = None,
) -> None:
    if not image_dir.is_dir():
        raise ValueError(f"Image directory {image_dir} does not exist or is not a directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", resolved_device)

    mask_generator = _build_mask_generator(model_cfg, checkpoint_path, resolved_device)
    refiner = _build_refiner(resolved_device) if use_postprocess else None

    image_id_map_by_name: dict = {}
    image_id_map_by_stem: dict = {}
    if image_id_map_path is not None:
        if not image_id_map_path.is_file():
            raise FileNotFoundError(f"Image id map {image_id_map_path} does not exist")
        image_id_map_by_name, image_id_map_by_stem = _load_image_id_map(image_id_map_path)

    image_paths = _collect_image_paths(image_dir, recursive)
    if not image_paths:
        logging.warning("No images found in %s", image_dir)
        return

    start_idx = start_index if start_index is not None else 1
    end_idx = end_index if end_index is not None else len(image_paths) + 1
    if start_idx < 1 or end_idx <= start_idx:
        raise ValueError("Invalid start/end indices")

    for idx, image_path in enumerate(image_paths, start=1):
        if idx < start_idx or idx >= end_idx:
            continue
        mapped_id = None
        if image_id_map_by_name or image_id_map_by_stem:
            mapped_id = image_id_map_by_name.get(image_path.name)
            if mapped_id is None:
                mapped_id = image_id_map_by_stem.get(image_path.stem)
        image_id = mapped_id if mapped_id is not None else _parse_image_id(image_path, idx)
        output_data = _annotate_image(
            image_path=image_path,
            image_id=image_id,
            mask_generator=mask_generator,
            granularities=granularities,
            use_postprocess=use_postprocess,
            refiner=refiner,
        )
        output_path = output_dir / f"{image_path.stem}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(output_data, fh, indent=2)
        logging.info("Wrote %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-image COCO annotations using SAM2.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory with input images")
    parser.add_argument("--model-config", type=str, required=True, help="Path to SAM2 model config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store per-image JSON files")
    parser.add_argument(
        "--granularities",
        type=float,
        nargs="+",
        default=[0.3, 0.7, 1.0],
        help="Granularity levels passed to SAM2 (default: 0.3 0.7 1.0)",
    )
    parser.add_argument("--device", type=str, default=None, help="Computation device (default: auto-detect)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for images")
    parser.add_argument(
        "--use-postprocess",
        action="store_true",
        help="Apply CascadePSP post-processing (requires segmentation_refinement)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument("--start-index", type=int, default=None, help="1-based index of first image to process")
    parser.add_argument("--end-index", type=int, default=None, help="1-based index of last image to process")
    parser.add_argument(
        "--image-id-map",
        type=Path,
        default=Path("/home/yujunwei/UnSAM/datasets/entity/val_lr_image_id_map.json"),
        help=(
            "Path to a JSON file mapping image names to ids. "
            "If provided, image ids in outputs follow the mapping and folder prefixes in the map are ignored."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s - %(levelname)s - %(message)s")
    run_annotation(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_cfg=args.model_config,
        checkpoint_path=args.checkpoint,
        granularities=args.granularities,
        recursive=args.recursive,
        use_postprocess=args.use_postprocess,
        device=args.device,
        start_index=args.start_index,
        end_index=args.end_index,
        image_id_map_path=args.image_id_map,
    )


if __name__ == "__main__":
    main()
