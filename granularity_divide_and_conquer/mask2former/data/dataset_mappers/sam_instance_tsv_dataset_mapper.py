# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import os
import numpy as np
import torch
import PIL.Image as Image
import json
import base64
from io import BytesIO

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import BitMasks, Instances, BoxMode

from pycocotools import mask as coco_mask
import cv2

__all__ = ["SamInstanceTSVDatasetMapper"]

_EXIF_ORIENT = 274
_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    image = _apply_exif_orientation(image)
    return convert_PIL_to_numpy(image, "RGB")


def cv2_img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    image = _apply_exif_orientation(image)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def area(mask):
    assert type(mask) is np.ndarray
    assert len(np.unique(mask)) <= 2
    if mask.size == 0: return 0
    return np.count_nonzero(mask) / mask.size

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        #T.ResizeScale(
        #     min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        # ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
        #T.ResizeShortestEdge(image_size)
    ])

    return augmentation


class SamInstanceTSVDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
        dataset_dir,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[SamInstanceTSVDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.dataset_dir = dataset_dir
    
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def tensor_to_cv2(self, image_tensor):
        """
        Convert a PyTorch tensor (C, H, W) to an OpenCV-compatible image (H, W, C) in BGR format.
        """
        if isinstance(image_tensor, torch.Tensor):  
            image_np = image_tensor.cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            image_np = (image_np * 255).astype(np.uint8)
            
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            return image_np
        else:
            raise TypeError("Input must be a PyTorch Tensor")
    def __call__(self, idx):
        """
        Args:
            dataset_path (str): path to the json file

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        tsv_dir= self.dataset_dir
        tsv_name = idx[0]
        lineidx = idx[1]

        with open(os.path.join(tsv_dir, tsv_name), 'r') as fp:
            fp.seek(lineidx)
            tsv_info = [s.strip() for s in fp.readline().split('\t')]

        dataset_dict = json.loads(tsv_info[1])
        image = cv2_img_from_base64(tsv_info[-1])

        # match the dataset_dict with model input
        dataset_dict["height"] = dataset_dict["image"]["height"]
        dataset_dict["width"] = dataset_dict["image"]["width"]
        dataset_dict["image_id"] = dataset_dict["image"]["id"]

        # match with default coco dataloader
        for ann in dataset_dict["annotations"]:
            ann["category_id"] = 0

        image_shape = image.shape[:2]
        dataset_dict["image"] = image

        return dataset_dict
