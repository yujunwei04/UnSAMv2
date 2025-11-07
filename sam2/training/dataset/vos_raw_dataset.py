# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch

from iopath.common.file_io import g_pathmgr

from omegaconf.listconfig import ListConfig

from training.dataset.vos_segment_loader import (
    JSONSegmentLoader,
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    UnSAMSegmentLoader,
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        additional_gt_folders=None,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.additional_gt_folders = []
        if additional_gt_folders:
            if isinstance(additional_gt_folders, (list, tuple)):
                candidate_folders = list(additional_gt_folders)
            else:
                candidate_folders = [additional_gt_folders]
            for folder in candidate_folders:
                if folder is None:
                    continue
                if not os.path.isdir(folder):
                    logging.warning(
                        f"Additional gt folder {folder} does not exist. Skipping."
                    )
                    continue
                self.additional_gt_folders.append(folder)
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for _, fpath in enumerate(all_frames[:: self.sample_rate]):
            fid = int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
    

class UnSAMRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        num_sa1b_videos=3000,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        num_frames=1,
        mask_area_frac_thresh=1.1,  # no filtering by default
        uncertain_iou=-1,  # no filtering by default
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.num_frames = num_frames
        self.mask_area_frac_thresh = mask_area_frac_thresh
        self.uncertain_iou = uncertain_iou  # stability score
        self.num_sa1b_videos = num_sa1b_videos

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.gt_folder)
            subset = [
                path.split(".")[0].replace("f_", "") for path in subset if path.endswith(".json")
            ]  # remove extension
        subset = subset[:6000] # change if want to use more data
        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        self.video_names = [
            video_name for video_name in subset if video_name not in excluded_files
        ]

        self.video_mask_paths = {}
        filtered_video_names = []
        for video_name in self.video_names:
            mask_paths = self._gather_mask_paths(video_name)
            if mask_paths is None:
                continue
            self.video_mask_paths[video_name] = mask_paths
            filtered_video_names.append(video_name)

        dropped_count = len(self.video_names) - len(filtered_video_names)
        if dropped_count > 0:
            logging.warning(
                f"Skipped {dropped_count} videos without masks present in all folders."
            )

        self.video_names = filtered_video_names
        self._num_primary_videos = len(self.video_names)
    
    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        if self.tsv_file and self.lineidx_file:
            video_name = self.video_names[idx]
            video_mask_path = os.path.join(self.gt_folder, "f_" + video_name + ".json")
            line_offset = self.idx_to_offset[idx]
            dataset_entry = self.mapper((os.path.basename(self.tsv_file), line_offset))
            image_data = dataset_entry["image"]
            image_data = image_data.copy()


            segment_loader = UnSAMSegmentLoader(
                video_mask_path=video_mask_path,
                mask_area_frac_thresh=self.mask_area_frac_thresh,
                video_frame_path=None,
                uncertain_iou=self.uncertain_iou,
                image_data=image_data,
            )
            frames = []
            tensor_data = torch.from_numpy(image_data.transpose(2, 0, 1)).float()

            for frame_idx in range(self.num_frames):
                frames.append(VOSFrame(frame_idx, image_path=None, data=tensor_data))
            video_name = video_name.split("_")[-1]  # filename is sa_{int}
            video = VOSVideo(video_name, int(video_name), frames)
            return video, segment_loader
        if self.sbd_gt_folder and self.sbd_img_folder and idx >= self.num_sa1b_videos:
            video_name = self.sbd_video_names[idx - self.num_sa1b_videos]
            video_frame_path = os.path.join(self.sbd_img_folder, video_name + ".jpg")
            video_mask_path = os.path.join(self.sbd_gt_folder, "f_" + video_name + ".json")

        else:
            video_name = self.video_names[idx]
            video_frame_path = os.path.join(self.img_folder, video_name + ".jpg")
            mask_paths = self.video_mask_paths.get(video_name)
            if mask_paths is None:
                resolved_mask = self._resolve_mask_path(self.gt_folder, video_name)
                if resolved_mask is None:
                    raise FileNotFoundError(
                        f"Could not locate mask json for {video_name} in primary or additional folders"
                    )
                mask_paths = [resolved_mask]
            video_mask_path = mask_paths
        segment_loader = UnSAMSegmentLoader(
            video_mask_path=video_mask_path,
            mask_area_frac_thresh=self.mask_area_frac_thresh,
            video_frame_path=video_frame_path,
            uncertain_iou=self.uncertain_iou,
        )

        frames = []
        for frame_idx in range(self.num_frames):
            frames.append(VOSFrame(frame_idx, image_path=video_frame_path))
        video_name = video_name.split("_")[-1]  # filename is sa_{int}
        video = VOSVideo(video_name, int(video_name), frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

    def _gather_mask_paths(self, video_name):
        mask_paths = []
        search_roots = [self.gt_folder] + list(self.additional_gt_folders)

        for root in search_roots:
            resolved_path = self._resolve_mask_path(root, video_name)
            if resolved_path is None:
                return None
            mask_paths.append(resolved_path)

        return mask_paths

    def _resolve_mask_path(self, folder, video_name):
        candidate_filenames = [f"{video_name}.json", f"f_{video_name}.json"]
        for candidate in candidate_filenames:
            mask_path = os.path.join(folder, candidate)
            if os.path.isfile(mask_path):
                return mask_path
        return None


class JSONRawDataset(VOSRawDataset):
    """
    Dataset where the annotation in the format of SA-V json files
    """

    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        rm_unannotated=True,
        ann_every=1,
        frames_fps=24,
    ):
        self.gt_folder = gt_folder
        self.img_folder = img_folder
        self.sample_rate = sample_rate
        self.rm_unannotated = rm_unannotated
        self.ann_every = ann_every
        self.frames_fps = frames_fps

        # Read and process excluded files if provided
        excluded_files = []
        if excluded_videos_list_txt is not None:
            if isinstance(excluded_videos_list_txt, str):
                excluded_videos_lists = [excluded_videos_list_txt]
            elif isinstance(excluded_videos_list_txt, ListConfig):
                excluded_videos_lists = list(excluded_videos_list_txt)
            else:
                raise NotImplementedError

            for excluded_videos_list_txt in excluded_videos_lists:
                with open(excluded_videos_list_txt, "r") as f:
                    excluded_files.extend(
                        [os.path.splitext(line.strip())[0] for line in f]
                    )
        excluded_files = set(excluded_files)

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, video_idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[video_idx]
        video_json_path = os.path.join(self.gt_folder, video_name + "_manual.json")
        segment_loader = JSONSegmentLoader(
            video_json_path=video_json_path,
            ann_every=self.ann_every,
            frames_fps=self.frames_fps,
        )

        frame_ids = [
            int(os.path.splitext(frame_name)[0])
            for frame_name in sorted(
                os.listdir(os.path.join(self.img_folder, video_name))
            )
        ]

        frames = [
            VOSFrame(
                frame_id,
                image_path=os.path.join(
                    self.img_folder, f"{video_name}/%05d.jpg" % (frame_id)
                ),
            )
            for frame_id in frame_ids[:: self.sample_rate]
        ]

        if self.rm_unannotated:
            # Eliminate the frames that have not been annotated
            valid_frame_ids = [
                i * segment_loader.ann_every
                for i, annot in enumerate(segment_loader.frame_annots)
                if annot is not None and None not in annot
            ]
            frames = [f for f in frames if f.frame_idx in valid_frame_ids]

        video = VOSVideo(video_name, video_idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)
