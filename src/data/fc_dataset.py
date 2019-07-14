import copy
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ext.histogram_matching import match_histograms


class FCDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, multiplier=1., transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms
        self.multiplier = multiplier
        self.clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        ribs_mean_brightness, clear_mean_brightness = 147, 150
        ribs_mean_range, clear_mean_range = 223, 222
        
        ribs_img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx]['ribs'])
        original_image = cv2.imread(ribs_img_name)[..., ::-1]
        clear_img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx]['cleared'])
        clear_image = cv2.imread(clear_img_name)[..., ::-1]
        
        total_image = np.zeros((clear_image.shape[0], clear_image.shape[1], 3), dtype=np.uint8)
        total_image[..., 0] = original_image[..., 0]
        total_image[..., 1:] = clear_image[..., :2]

        if self.transform:
            augmented = self.transform(image=total_image)
            total_image = augmented['image']

        original_image = np.expand_dims(self.clahe.apply(total_image[..., 0].astype(np.uint8)), axis=2)
        clear_image = np.expand_dims(self.clahe.apply(total_image[..., 1].astype(np.uint8)), axis=2)

        original_image = torch.from_numpy(np.moveaxis((np.expand_dims(original_image[..., 0], axis=2) / 255.), -1, 0).astype(np.float32))

        clear_image = torch.from_numpy(np.moveaxis((np.expand_dims(clear_image[..., 0], axis=2) / 255.), -1, 0).astype(np.float32))

        sample = {'input': original_image, 
                  'target': clear_image}
        return sample
