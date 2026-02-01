"""
Written by Yian Zhao
"""

from pathlib import Path
import pickle as pkl
import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import random
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

# Lazy import for clip (only needed for training, not evaluation)
_clip = None
def _get_clip():
    global _clip
    if _clip is None:
        from isegm.model.modeling.clip import clip
        _clip = clip
    return _clip

class PascalPartDataset(ISDataset):

    def __init__(self, dataset_path, split='train', **kwargs):
        super(PascalPartDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'JPEGImages'
        self._anno_path = self.dataset_path / 'Annotations_Part'

        with open(self.dataset_path / f'ImageSets/Main/{self.dataset_split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

        self.abbr2word = {
            "lwing": "left wing", "rwing": "right wing", "fwheel": "front wheel", "bwheel": "back wheel",
            "haddlebar": "haddle bar", "chainwheel": "chain wheel",
            "leye": "left eye", "reye": "right eye", "lwing": "left wing", "rwing": "right wing", "lleg": "left leg",
            "rleg": "right leg", "lfoot": "left foot",
            "rfoot": "right foot", "fliplate": "front license plate", "bliplate": "back license plate",
            "lear": "left ear", "rear": "right ear", "lfleg": "left front leg",
            "lfpa": "left front paw", "rfleg": "right front leg", "rfpa": "right front paw", "lbleg": "left back leg",
            "lbpa": "left back paw", "rbleg": "right back leg",
            "rbpa": "right back paw", "lhorn": "left horn", "rhorn": "right horn", "lfuleg": "left front upper leg",
            "lflleg": "left front lower leg", "rfuleg": "right front upper leg",
            "rflleg": "right front lower leg", "lbuleg": "left back upper leg", "lblleg": "left back lower leg",
            "rbuleg": "right back upper leg", "rblleg": "right back lower leg",
            "lfho": "left front hoof", "rfho": "right front hoof", "lbho": "left back hoof", "rbho": "right back hoof",
            "lebrow": "left eyebrow", "rebrow": "right eyebrow",
            "llarm": "left lower arm", "luarm": "left upper arm", "rlarm": "right lower arm",
            "ruarm": "right upper arm", "lhand": "left hand", "rhand": "right hand", "llleg": "left lower leg",
            "luleg": "left upper leg", "rlleg": "right lower leg", "ruleg": "right upper leg", "lfoot": "left foot",
            "rfoot": "right foot", "hfrontside": "head front side",
            "hleftside": "head left side", "hrightside": "head right side", "hbackside": "head back side",
            "hroofside": "head roof side", "cfrontside": "coach front side",
            "cleftside": "coach left side", "crightside": "coach right side", "cbackside": "coach back side",
            "croofside": "coach roof side", "leftmirror": "left mirror",
            "rightmirror": "right mirror", "frontside": "front side", "backside": "back side", "leftside": "left side",
            "rightside": "right side", "roofside": "roof side",
        }
        self.probs = {
            0.2: 0.06960800373840834, 0.3: 0.09004898499961728, 1.0: 0.06696065989491415, 0.1: 0.012166262603621974,
            0.4: 0.10218302562007273,
            0.6: 0.12601478544277883, 0.5: 0.11521376884572793, 0.7: 0.1343502368084108, 0.9: 0.14037031069344932,
            0.8: 0.14308396135299856
        }  # precompute
        if self.probs is None:
            self._count_gra()

    def _count_gra(self):
        gra_cnt = {}
        for image_name in tqdm(self.dataset_samples):
            inst_info_path = str(self._anno_path / f'{image_name}.mat')
            ann = loadmat(inst_info_path)['anno']['objects'][0, 0]
            for instance_id in range(ann.shape[1]):
                parts = ann['parts'][0, instance_id]
                masks = [PartNode(ann[0, instance_id]['mask'], ann['class'][0, instance_id][0])]
                for part_id in range(parts.shape[1]):
                    masks.append(PartNode(parts['mask'][0, part_id], self._prase_part_name(parts['part_name'][0, part_id][0])))
                # calculate gra for each mask
                gras = self._get_gra_list(masks, ann[0, instance_id]['mask'])
                for gra in gras:
                    gra_cnt[gra] = gra_cnt.get(gra, 0) + 1

        gra_probs = {}
        total_cnt = sum(gra_cnt.values())
        for gra, cnt in gra_cnt.items():
            gra_probs[gra] = np.log(total_cnt / cnt)

        self.probs = {gra: x / sum(gra_probs.values()) for gra, x in gra_probs.items()}

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._anno_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann = loadmat(inst_info_path)['anno']['objects'][0, 0]

        instance_id = np.random.randint(0, ann.shape[1])
        class_name = ann['class'][0, instance_id][0]

        parts = ann['parts'][0, instance_id]
        if parts.shape[1] > 0:
            masks = [PartNode(ann[0, instance_id]['mask'], class_name)]
            for part_id in range(parts.shape[1]):
                masks.append(PartNode(parts['mask'][0, part_id], self._prase_part_name(parts['part_name'][0, part_id][0])))
            # calculate gra for each mask
            gras = self._get_gra_list(masks, ann[0, instance_id]['mask'])
            # calculate probs for each mask
            probs = self._get_prob_list(gras)
            tgt_idx = np.random.choice(range(len(masks)), p=probs)
            instances_mask = masks[tgt_idx].mask
            part_name = masks[tgt_idx].name
            gra = gras[tgt_idx]
        else:
            # part == instance
            instances_mask = ann[0, instance_id]['mask']
            part_name = class_name
            gra = 1.0

        if part_name is not None:
            clip = _get_clip()
            prompt = clip.tokenize(part_name).squeeze(0)

        return DSample(image, instances_mask, gra=gra, prompt=prompt, objects_ids=[1], sample_id=index)

    def _get_gra_list(self, masks, obj_mask):
        obj_area = np.bincount(obj_mask.flatten())[1]
        gras = []
        for node in masks:
            part_mask = node.mask
            part_area = np.bincount(part_mask.flatten())[1]
            gras.append(max(round(part_area / obj_area, 1), 0.1))

        return gras

    def _get_prob_list(self, gras):
        gra_num_each, probs_nouni = {}, {}
        for gra in gras:
            gra_num_each[gra] = gra_num_each.get(gra, 0) + 1

        for gra, cnt in gra_num_each.items():
            probs_nouni[gra] = self.probs[gra] / cnt
        p = []
        for gra in gras:
            p.append(probs_nouni[gra])
        sum_p = sum(p)
        p = [x / sum_p for x in p]
        return p

    def _prase_part_name(self, part_name):
        part_name = part_name.split("_")[0]
        if part_name in self.abbr2word.keys():
            part_name = self.abbr2word[part_name]

        return part_name

class PartNode:
    def __init__(self, mask, name):
        self.mask = mask
        self.name = name

    def __repr__(self):
        return f"{self.name}"


class PascalPartEvaluationDataset(ISDataset):

    def __init__(self, dataset_path, split='val', class_name=None, part_name=None, **kwargs):
        super(PascalPartEvaluationDataset, self).__init__(**kwargs)
        assert split in {'train', 'val'}

        self.dataset_path = Path(dataset_path)
        self.dataset_split = split
        self._images_path = self.dataset_path / 'JPEGImages'
        self._anno_path = self.dataset_path / 'Annotations_Part'
        with open(self.dataset_path / f'ImageSets/Main/{self.dataset_split}.txt', 'r') as f:
            self.dataset_samples = [name.strip() for name in f.readlines()]

        self.all_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',  'train', 'tvmonitor']
        if class_name is not None:
            assert class_name in self.all_class
        self.dataset_samples = self.get_sbd_images_and_ids_list(class_name, part_name)

    def get_sample(self, index) -> DSample:
        image_name, instance_id, part_id, class_name, part_name = self.dataset_samples[index]
        image_path = str(self._images_path / f'{image_name}.jpg')
        inst_info_path = str(self._anno_path / f'{image_name}.mat')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = loadmat(str(inst_info_path))['anno']['objects'][0, 0]['parts'][0, instance_id]['mask'][0, part_id]

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

    def get_sbd_images_and_ids_list(self, class_name=None, part_name=None):
        if class_name is not None and part_name is not None:
            pkl_path = self.dataset_path / f'{class_name}_{part_name}_{self.dataset_split}_images_and_ids_list.pkl'
        elif part_name is not None:
            pkl_path = self.dataset_path / f'{part_name}_{self.dataset_split}_images_and_ids_list.pkl'
        elif class_name is not None:
            pkl_path = self.dataset_path / f'{class_name}_{self.dataset_split}_images_and_ids_list.pkl'
        else:
            pkl_path = self.dataset_path / f'{self.dataset_split}_images_and_ids_list.pkl'

        if pkl_path.exists():
            with open(str(pkl_path), 'rb') as fp:
                images_and_ids_list = pkl.load(fp)
        else:
            # TODO: allow class list eval
            images_and_ids_list = []
            for sample in self.dataset_samples:
                inst_info_path = str(self._anno_path / f'{sample}.mat')
                ann = loadmat(inst_info_path)['anno']['objects'][0, 0]
                keep_ann = (class_name is None) and (part_name is not None)
                n_objects = ann.shape[1]
                for i in range(n_objects):
                    if not keep_ann and class_name is not None and ann['class'][0, i][0] != class_name:
                        continue
                    parts = ann['parts'][0, i]
                    n_parts = parts.shape[1]
                    for j in range(n_parts):
                        if part_name is not None and parts['part_name'][0, j][0].split("_")[0] != part_name:
                            continue
                        images_and_ids_list.append((sample, i, j, ann['class'][0, i][0], parts['part_name'][0, j][0]))

            if class_name is None and part_name is None:
                images_and_ids_list = random.sample(images_and_ids_list, int(len(images_and_ids_list) / 10))
        
            with open(str(pkl_path), 'wb') as fp:
                pkl.dump(images_and_ids_list, fp)
        
        return images_and_ids_list