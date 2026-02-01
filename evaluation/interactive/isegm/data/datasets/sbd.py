import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from scipy.io import loadmat
from isegm.utils.misc import get_bbox_from_mask, get_labels_with_sizes
from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class SBDDataset(ISDataset):
    def __init__(self, dataset_path, split='train', buggy_mask_thresh=0.08, partmask_path=None, 
                 disable_probs=False, **kwargs):
        super(SBDDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._insts_path = self.dataset_path / 'inst'
        self._buggy_objects = dict()
        self._buggy_mask_thresh = buggy_mask_thresh
        self.disable_probs = disable_probs

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]
        
        self.partmask_path = partmask_path
        if partmask_path is not None:
            with open(partmask_path, 'rb') as f:
                self.partmask_samples = pkl.load(f)

            self.gra_probs = self._count_gra(self.partmask_samples)

    def _count_gra(self, partmask_samples):
        gra_num = {}
        for img_name in partmask_samples.keys():
            gras = partmask_samples[img_name]["gra"]
            for idx in range(gras.shape[0]):
                gra = gras[idx][1]
                gra_num[gra] = gra_num.get(gra, 0) + 1
        
        sum_num = 0
        for k, v in gra_num.items():
            if k != 0.0:
                sum_num += v

        gra_probs = {}
        for k, v in gra_num.items():
            if k != 0.0:
                gra_probs[k] = sum_num / v
        gra_probs[0.0] = 0.

        return gra_probs
        
    def get_sample(self, index):
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.partmask_path is not None and image_name in self.partmask_samples.keys():
            part_data = self.partmask_samples[str(image_name)]
            instances_mask, gra = part_data['mask'], part_data['gra']
            
            if isinstance(gra[0], np.ndarray):
                gra = np.array([round(gs[0] * 0.1 + gs[1] * 0.9, 1) if gs[0] <= 1.0 and abs(gs[1] - gs[0]) < 0.1 else gs[1] for gs in gra])

            if not self.disable_probs:
                gra_num_each, probs = {}, {}
                for g in gra:
                    gra_num_each[g] = gra_num_each.get(g, 0) + 1
                for g, cnt in gra_num_each.items():
                    probs[g] = self.gra_probs[g] / cnt
                p = []
                for g in gra:
                    p.append(probs[g])
                sum_p = sum(p)
                p = [x / sum_p for x in p]
                tgt_idx = np.random.choice(range(gra.shape[0]), p=p)
                instances_mask, gra = instances_mask[tgt_idx], gra[tgt_idx]
            else:
                tgt_idx = np.random.choice(range(gra.shape[0]))
                instances_mask, gra = instances_mask[tgt_idx], gra[tgt_idx]

        else:
            instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
            gra = 1.0

        instances_mask = self.remove_buggy_masks(index, instances_mask)
        instances_ids, _ = get_labels_with_sizes(instances_mask)

        return DSample(image, instances_mask, gra=gra, objects_ids=instances_ids, sample_id=index)

    def remove_buggy_masks(self, index, instances_mask):
        if self._buggy_mask_thresh > 0.0:
            buggy_image_objects = self._buggy_objects.get(index, None)
            if buggy_image_objects is None:
                buggy_image_objects = []
                instances_ids, _ = get_labels_with_sizes(instances_mask)
                for obj_id in instances_ids:
                    obj_mask = instances_mask == obj_id
                    mask_area = obj_mask.sum()
                    bbox = get_bbox_from_mask(obj_mask)
                    bbox_area = (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)
                    obj_area_ratio = mask_area / bbox_area
                    if obj_area_ratio < self._buggy_mask_thresh:
                        buggy_image_objects.append(obj_id)

                self._buggy_objects[index] = buggy_image_objects
            for obj_id in buggy_image_objects:
                instances_mask[instances_mask == obj_id] = 0

        return instances_mask


class SBDEvaluationDataset(ISDataset):
    def __init__(self, dataset_path, split='val', **kwargs):
        super(SBDEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'img'
        self._insts_path = self.dataset_path / 'inst'

        with open(self.dataset_path / f'{split}.txt', 'r') as f:
            self.dataset_samples = [x.strip() for x in f.readlines()]
        self.dataset_samples = self.get_sbd_images_and_ids_list()

    def get_sample(self, index) -> DSample:
        image_name, instance_id = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._insts_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
        instances_mask[instances_mask != instance_id] = 0
        instances_mask[instances_mask > 0] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

    def get_sbd_images_and_ids_list(self):
        pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            images_and_ids_list = []

            for sample in self.dataset_samples:
                inst_info_path = str(self._insts_path / f'{sample}.mat')
                instances_mask = loadmat(str(inst_info_path))['GTinst'][0][0][0].astype(np.int32)
                instances_ids, _ = get_labels_with_sizes(instances_mask)

                for instances_id in instances_ids:
                    images_and_ids_list.append((sample, instances_id))

            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)

        return images_and_ids_list
