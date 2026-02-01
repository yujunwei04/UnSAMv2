"""
Written by Yian Zhao
"""

from pycocotools import mask as maskUtils
import os, json, cv2
import numpy as np
from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import pickle as pkl
from pathlib import Path
import random
import tqdm
import torch

def load_sample(root, id, annotation_dir='annotations', image_dir='images'):
    img = os.path.join(root, image_dir, id + ".jpg")
    annotation = os.path.join(root, annotation_dir, id + ".json")
    return (img, annotation)

class SA1BDataset(ISDataset):
    """A class to load SA-1B: https://segment-anything.com/dataset/index.html

    Attributes:
        min_object (int): The minimum number of pixels required for an object to be considered valid.
        samples (list): A list of loaded samples from the dataset.
        image_info (list): A list containing image information for each sample.

    """

    def __init__(self, dataset_dir, ids=None, annotation_dir='annotations', image_dir='images', min_object=1,
                 num_images=1000, **kwargs):
        """Initializes SA1BDataset class.

        Args:
            dataset_dir (str): The directory containing the dataset.
            ids (list, optional): A list of sample IDs to load. If not provided, the class
                                  will load all samples found in the annotation_dir. Default is None.
            annotation_dir (str, optional): The directory containing annotation files
                                            (relative to dataset_dir). Default is 'annotations'.
            image_dir (str, optional): The directory containing image files
                                       (relative to dataset_dir). Default is 'images'.
            min_object (int, optional): The minimum number of pixels required for an object
                                        to be considered valid. Default is 0.
            num_images (int, optional): The number of images to load. Default is 1000.
        """
        super(SA1BDataset, self).__init__(**kwargs)
        if ids is None:
            ids = [file.replace(".json", '') for file in os.listdir(os.path.join(dataset_dir, annotation_dir))]
        self.min_object = min_object
        self.samples = [load_sample(dataset_dir, id, annotation_dir, image_dir) for id in ids][:num_images]
        self.dataset_dir = Path(dataset_dir)

        self.dataset_samples = self.get_images_and_ids_list()

    def get_sample(self, index):
        image_path, annotation, h, w = self.dataset_samples[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        instance_mask = self.annToMask(annotation, h, w)

        return DSample(image, instance_mask, objects_ids=[1], sample_id=index)
    
    

    def get_images_and_ids_list(self):
        pkl_path = self.dataset_dir / "images_and_ids_list.pkl"

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0

        images_and_ids_list = None
        if rank == 0:
            if pkl_path.exists():
                with open(str(pkl_path), 'rb') as fp:
                    images_and_ids_list = pkl.load(fp)
            else:
                images_and_ids_list = []
                for img, info in tqdm.tqdm(self.samples):
                    image_info = json.load(open(info))
                    annotations = image_info["annotations"]
                    image_info = image_info['image']

                    instance_masks = []
                    for annotation in annotations:
                        instance_mask = self.annToMask(annotation, image_info["height"],
                                                       image_info["width"])
                        # Some objects are so small that they're less than 1 pixel area
                        # and end up rounded out. Skip those objects.
                        area = instance_mask.sum()
                        if area < self.min_object:
                            continue
                        instance_masks.append(instance_mask)

                    # random sample max five for each image
                    selected_masks = []
                    selected_indices = []
                    max_len = min(len(instance_masks), 5)
                    max_iter = len(instance_masks) * 2
                    cur_iter = 0
                    while len(selected_masks) < max_len and cur_iter < max_iter:
                        mask_index = random.randint(0, len(instance_masks) - 1)
                        instance_mask = instance_masks[mask_index]
                        is_overlap = False
                        for selected_mask in selected_masks:
                            if np.any(np.logical_and(instance_mask, selected_mask)):
                                is_overlap = True
                                break
                        if not is_overlap:
                            selected_indices.append(mask_index)
                            selected_masks.append(instance_mask)
                            instance_masks.pop(mask_index)
                        cur_iter += 1

                    for idx in selected_indices:
                        images_and_ids_list.append((img, annotations[idx], image_info["height"], image_info["width"]))

                with open(str(pkl_path), 'wb') as fp:
                    pkl.dump(images_and_ids_list, fp)
        if is_distributed:
            obj_list = [images_and_ids_list]
            torch.distributed.broadcast_object_list(obj_list, src=0)
            images_and_ids_list = obj_list[0]

        return images_and_ids_list

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
