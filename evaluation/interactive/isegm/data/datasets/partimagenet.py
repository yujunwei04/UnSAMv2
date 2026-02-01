"""
Written by Yian Zhao
"""

from pathlib import Path
import pickle as pkl
import cv2
import os
import numpy as np

from isegm.utils.misc import get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

class PartINEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='val', images_dir_name='images', masks_dir_name='annotations',
                 class_name=None, part_name=None, **kwargs):
        super(PartINEvaluationDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / images_dir_name / split
        self._insts_path = self.dataset_path / masks_dir_name / split
        self._class_names = {}
        with open(os.path.join(self.dataset_path, "class_names.json"), "r") as f:
            names = eval(f.read())["categories"]
        for item in names:
            self._class_names[item["id"]] = item["name"]

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.*'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.png')}
        self.dataset_samples = self.get_images_and_ids_list(class_name, part_name)

    def get_sample(self, index) -> DSample:
        image_name, instance_id = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)
        mask_path = str(self._masks_paths[image_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path).astype(np.int32)[:, :, 0]
        instances_mask[instances_mask != instance_id] = 0
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

    def get_images_and_ids_list(self, class_name=None, part_name=None):
        if class_name is not None and part_name is not None:
            pkl_path = self.dataset_path / f'{class_name}_{part_name}_{self.dataset_split}_images_and_ids_list.pkl'
        elif part_name is not None:
            pkl_path = self.dataset_path / f'{part_name}_{self.dataset_split}_images_and_ids_list.pkl'
        elif class_name is not None:
            pkl_path = self.dataset_path / f'{class_name}_{self.dataset_split}_images_and_ids_list.pkl'
        else:
            pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list_include_40.pkl'

        images_and_ids_list = []

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list.extend(pkl.load(fp))
        else:
            # TODO: allow class list eval
            for sample in self.dataset_samples:
                mask_path = str(self._masks_paths[sample.split('.')[0]])
                instances_mask = cv2.imread(mask_path).astype(np.int32)
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                # instances_ids.remove(40)  # 40 is background
                keep_ann = (class_name is None) and (part_name is not None)
                for instances_id in instances_ids:
                    if not keep_ann and class_name is not None and self._class_names[instances_id].split()[
                        0].lower() != class_name:
                        continue
                    if part_name is not None and self._class_names[instances_id].split()[1].lower() != part_name:
                        continue
                    images_and_ids_list.append((sample, instances_id))

            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list