# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import logging
import os

import numpy as np
import pandas as pd
import torch


from PIL import Image as PILImage

from sam2.modeling.sam2_utils import (
    get_1d_sine_pe,
    get_next_point,
    sample_box_points,
    select_closest_cond_frames,
)

try:
    from pycocotools import mask as mask_utils
except:
    pass


class JSONSegmentLoader:
    def __init__(self, video_json_path, ann_every=1, frames_fps=24, valid_obj_ids=None):
        # Annotations in the json are provided every ann_every th frame
        self.ann_every = ann_every
        # Ids of the objects to consider when sampling this video
        self.valid_obj_ids = valid_obj_ids
        with open(video_json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                self.frame_annots = data
            elif isinstance(data, dict):
                masklet_field_name = "masklet" if "masklet" in data else "masks"
                self.frame_annots = data[masklet_field_name]
                if "fps" in data:
                    if isinstance(data["fps"], list):
                        annotations_fps = int(data["fps"][0])
                    else:
                        annotations_fps = int(data["fps"])
                    assert frames_fps % annotations_fps == 0
                    self.ann_every = frames_fps // annotations_fps
            else:
                raise NotImplementedError

    def load(self, frame_id, obj_ids=None):
        assert frame_id % self.ann_every == 0
        rle_mask = self.frame_annots[frame_id // self.ann_every]

        valid_objs_ids = set(range(len(rle_mask)))
        if self.valid_obj_ids is not None:
            # Remove the masklets that have been filtered out for this video
            valid_objs_ids &= set(self.valid_obj_ids)
        if obj_ids is not None:
            # Only keep the objects that have been sampled
            valid_objs_ids &= set(obj_ids)
        valid_objs_ids = sorted(list(valid_objs_ids))

        # Construct rle_masks_filtered that only contains the rle masks we are interested in
        id_2_idx = {}
        rle_mask_filtered = []
        for obj_id in valid_objs_ids:
            if rle_mask[obj_id] is not None:
                id_2_idx[obj_id] = len(rle_mask_filtered)
                rle_mask_filtered.append(rle_mask[obj_id])
            else:
                id_2_idx[obj_id] = None

        # Decode the masks
        raw_segments = torch.from_numpy(mask_utils.decode(rle_mask_filtered)).permute(
            2, 0, 1
        )  # （num_obj, h, w）
        segments = {}
        for obj_id in valid_objs_ids:
            if id_2_idx[obj_id] is None:
                segments[obj_id] = None
            else:
                idx = id_2_idx[obj_id]
                segments[obj_id] = raw_segments[idx]
        return segments

    def get_valid_obj_frames_ids(self, num_frames_min=None):
        # For each object, find all the frames with a valid (not None) mask
        num_objects = len(self.frame_annots[0])

        # The result dict associates each obj_id with the id of its valid frames
        res = {obj_id: [] for obj_id in range(num_objects)}

        for annot_idx, annot in enumerate(self.frame_annots):
            for obj_id in range(num_objects):
                if annot[obj_id] is not None:
                    res[obj_id].append(int(annot_idx * self.ann_every))

        if num_frames_min is not None:
            # Remove masklets that have less than num_frames_min valid masks
            for obj_id, valid_frames in list(res.items()):
                if len(valid_frames) < num_frames_min:
                    res.pop(obj_id)

        return res


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        video_png_root: the folder contains all the masks stored in png
        """
        self.video_png_root = video_png_root
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        png_filenames = os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for filename in png_filenames:
            frame_id, _ = os.path.splitext(filename)
            self.frame_id_to_png_filename[int(frame_id)] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # load the mask
        masks = PILImage.open(mask_path).convert("P")
        masks = np.array(masks)

        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # remove background (0)

        # convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]
        self.W = tmp_mask.shape[1]
        if self.single_object_mode:
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class LazySegments:
    """
    Only decodes segments that are actually used.
    """

    def __init__(self):
        self.segments = {}
        self.cache = {}
        self.granularity_scores = {}
        self.divide_masks = set()

    def __setitem__(self, key, item):
        self.segments[key] = item

    def set_granularity(self, key, score=1.0):
        self.granularity_scores[key] = score

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]

        item = self.segments[key]
        rle = item[0]
        mask = torch.from_numpy(mask_utils.decode([rle])).permute(2, 0, 1)[0]
        self.cache[key] = mask
        return mask

    def get_granularity(self, key):
        return self.segments[key][1]

    def __contains__(self, key):
        return key in self.segments

    def __len__(self):
        return len(self.segments)

    def keys(self):
        return self.segments.keys()


class SA1BSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
    ):
        with open(video_mask_path, "r") as f:
            self.frame_annots = json.load(f)

        if mask_area_frac_thresh <= 1.0:
            # Lazily read frame
            orig_w, orig_h = PILImage.open(video_frame_path).size
            area = orig_w * orig_h

        self.frame_annots = self.frame_annots["annotations"]

        rle_masks = []
        granularity_scores = []
        granularity_prev_scores = []
        for frame_annot in self.frame_annots:
            if not frame_annot["area"] > 0:
                continue
            if ("uncertain_iou" in frame_annot) and (
                frame_annot["uncertain_iou"] < uncertain_iou
            ):
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (frame_annot["area"] / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])
            if 'granularity' in frame_annot.keys():
                granularity_scores.append(frame_annot['granularity'])
            else:
                granularity_scores.append(1.0)
            # prev granularity (fallback to current if missing)
            if 'granularity_prev' in frame_annot.keys():
                granularity_prev_scores.append(frame_annot['granularity_prev'])
            else:
                granularity_prev_scores.append(granularity_scores[-1])

        self.segments = LazySegments()
        for i, triple in enumerate(zip(rle_masks, granularity_scores, granularity_prev_scores)):
            self.segments[i] = triple

    def compute_granularity_score(self, frame_annot):
        """
        Compute granularity score for a given frame annotation.
        Currently, it returns a constant 1.0.
        """
        return 1.0


    def load(self, frame_idx):
        # return self.segments
        return self.segments

class UnSAMSegmentLoader:
    def __init__(
        self,
        video_mask_path,
        mask_area_frac_thresh=1.1,
        video_frame_path=None,
        uncertain_iou=-1,
        image_data=None,
    ):
        mask_paths = self._normalize_mask_paths(video_mask_path)
        if len(mask_paths) == 0:
            raise ValueError("No mask json paths provided to UnSAMSegmentLoader")

        annotations = []
        for path in mask_paths:
            with open(path, "r") as f:
                mask_data = json.load(f)
            if not isinstance(mask_data, dict) or "annotations" not in mask_data:
                raise ValueError(
                    f"Mask file {path} does not contain expected 'annotations' field"
                )
            annotations.extend(mask_data["annotations"])

        self.frame_annots = annotations

        if mask_area_frac_thresh <= 1.0:
            if image_data is None:
                orig_w, orig_h = PILImage.open(video_frame_path).size
            else:
                orig_h, orig_w = image_data.shape[:2]
            area = orig_w * orig_h

        rle_masks = []
        granularity_scores = []
        granularity_prev_scores = []
        self.segments = LazySegments()
        granularity_buckets = self.segments.granularity_buckets
        divide_masks = self.segments.divide_masks

        for mask_idx, frame_annot in enumerate(self.frame_annots):
            m = mask_utils.decode(frame_annot["segmentation"])
            m_tensor = torch.from_numpy(m).long()
            if not self.area(m) > 0:
                continue
            if (
                mask_area_frac_thresh <= 1.0
                and (self.area(m) / area) >= mask_area_frac_thresh
            ):
                continue
            rle_masks.append(frame_annot["segmentation"])
            new_idx = len(rle_masks) - 1
            gra = 1.0

            if 'is_divide' in frame_annot.keys() and frame_annot['is_divide']:
                divide_masks.add(new_idx)

            if 'granularity' in frame_annot.keys():
                if isinstance(frame_annot['granularity'], (float, int)):
                    if not np.isnan(frame_annot['granularity']) and not np.isinf(frame_annot['granularity']):
                        granularity_scores.append(frame_annot['granularity'])
                        gra = frame_annot['granularity']
                    else:
                        logging.warning(f"Found NaN or Inf in granularity, using default value 1.0")
                        granularity_scores.append(1.0)
                        gra = 1.0
                else:
                    logging.warning(f"Granularity is not a number: {type(frame_annot['granularity'])}, using default value 1.0")
                    granularity_scores.append(1.0)
                    gra = 1.0
            else:
                granularity_scores.append(1.0)
                gra = 1.0

        for i, pair in enumerate(zip(rle_masks, granularity_scores)):
            self.segments[i] = pair

    def area(self, mask):
        return np.count_nonzero(mask) / mask.size

    def load(self, frame_idx):
        return self.segments

    @staticmethod
    def _normalize_mask_paths(mask_paths):
        if mask_paths is None:
            return []
        if isinstance(mask_paths, (list, tuple)):
            return [mp for mp in mask_paths if mp]
        return [mask_paths]
