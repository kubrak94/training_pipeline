import copy
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ext.histogram_matching import match_histograms


class AlbumentationsDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
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

        if self.transform:
            seed = random.randint(0,2**32)  # so both of the images will be transformed the same way
            random.seed(seed)
            augmented = self.transform(image=original_image)
            original_image = augmented['image']
            random.seed(seed)
            augmented = self.transform(image=clear_image)
            clear_image = augmented['image']
            
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        clear_image = cv2.cvtColor(clear_image, cv2.COLOR_RGB2GRAY)

        #original_image = cv2.equalizeHist(original_image)
        #clear_image = cv2.equalizeHist(clear_image)

        original_image = self.clahe.apply(original_image)
        clear_image = self.clahe.apply(clear_image)
        
        original_image = original_image / 255.
        clear_image = clear_image / 255.
        
        original_image = torch.Tensor(original_image.reshape(1,576,576))
        clear_image = torch.Tensor(clear_image.reshape(1,576,576))

        sample = {'ribs': original_image, 
                  'cleared': clear_image}
        return sample
