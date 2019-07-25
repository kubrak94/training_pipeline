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
        # change CLAHE to the augmentation of albumentations
        self.clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        
        ribs_img_name = os.path.join(self.root_dir,
                                     self.csv_file.iloc[idx]['ribs'])
        original_image = cv2.imread(ribs_img_name)[..., ::-1]
        clear_img_name = os.path.join(self.root_dir,
                                      self.csv_file.iloc[idx]['cleared'])
        clear_image = cv2.imread(clear_img_name)[..., ::-1]
        
        #original_image = self.clahe.apply(original_image.astype(np.uint8))
        #clear_image = self.clahe.apply(clear_image.astype(np.uint8))

        if self.transform:
            augmented = self.transform(image=original_image, image2=clear_image)
            original_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2GRAY)
            clear_image = cv2.cvtColor(augmented['image2'], cv2.COLOR_RGB2GRAY)

        # TODO: make normalization and std parameters of config file
        original_image = (original_image / 255. - 0.5) / 0.25
        original_image = np.expand_dims(original_image, axis=2)
        clear_image = np.expand_dims(clear_image, axis=2)
        
        original_image = np.moveaxis((original_image), -1, 0)
        clear_image = np.moveaxis((clear_image), -1, 0)

        original_image = torch.from_numpy(original_image).type(torch.float32)
        clear_image = torch.from_numpy(clear_image).type(torch.float32)

        sample = {'input': original_image, 
                  'target': clear_image}
        return sample
