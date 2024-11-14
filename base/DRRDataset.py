import cv2
import logging
import numpy as np
import os
import random
import sys
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.abspath(".."))
from base.Utils import *

DATASET_META_FILENAME = "dataset.meta"

Logger = logging.getLogger('global')

"""DRR Dataset."""
class DRRDataset(Dataset):

    def __init__(self, dataset_dir, metadata, configs, transform=None):
        super(DRRDataset, self).__init__()
        self.metadata = metadata
        self.dataset_dir = dataset_dir
        self.configs = configs

    def __len__(self):
        return len(self.metadata) - 1
    
    def generate_number(self, idx):
        while True:
            random_number = random.randint(1, 30)
            if idx + random_number <= self.__len__():
                return random_number

    def __getitem__(self, idx):
        template_image_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx, 1])
        template_mask_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx, 2])
        offset = self.generate_number(idx)
        search_image_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx + offset, 1])
        search_mask_path = os.path.join(self.dataset_dir, self.metadata.iloc[idx + offset, 2])

        template_image = cv2.imread(template_image_path, cv2.IMREAD_GRAYSCALE)
        template_mask = cv2.imread(template_mask_path, cv2.IMREAD_GRAYSCALE)
        search_image = cv2.imread(search_image_path, cv2.IMREAD_GRAYSCALE)
        search_mask = cv2.imread(search_mask_path, cv2.IMREAD_GRAYSCALE)

        # search_size, template_size = crop_sizes(template_mask)
        # template_image = crop(template_image, template_mask, template_size)
        # template_mask = crop(template_mask, template_mask, template_size)
        # search_image = crop(search_image, search_mask, search_size)
        # search_mask = crop(search_mask, search_mask, search_size)

        template_image = cv2.resize(template_image, (self.configs.template_size[0], self.configs.template_size[1]), cv2.INTER_LINEAR)
        template_mask = cv2.resize(template_mask, (self.configs.template_size[0], self.configs.template_size[1]), cv2.INTER_LINEAR)
        search_image = cv2.resize(search_image, (self.configs.search_size[0], self.configs.search_size[1]), cv2.INTER_LINEAR)
        search_mask = cv2.resize(search_mask, (self.configs.search_size[0], self.configs.search_size[1]), cv2.INTER_LINEAR)

        template_image = cv2.normalize(template_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16F)
        template_mask[template_mask < 255.0] = 0.0
        template_mask[template_mask == 255.0] = 1.0
        template_mask = 1 - template_mask

        search_image = cv2.normalize(search_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_16F)
        search_mask[search_mask < 255.0] = 0.0
        search_mask[search_mask == 255.0] = 1.0
        search_mask = 1 - search_mask

        image = np.stack([search_image, template_mask, template_image], axis=0)

        return {
            'image': image,
            'template_image': template_image,
            'template_mask': template_mask,
            'search_image': search_image,
            'search_mask': search_mask
        }