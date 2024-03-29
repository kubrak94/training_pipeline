import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform.histogram_matching import match_histograms
from src.train.transforms import Normalize, ToGray


class FCDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, repeat_dataset=1,
                 mean=0.5, std=0.25, max_pixel_value=1.0, 
                 transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            repeat_dataset (int): how many times copy content of the csv_file
            mean (float, float, float): mean values for image normalization
            std  (float, float, float): std values for image normalization
            max_pixel_value (float): maximum possible pixel value
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        _csv_file = pd.read_csv(csv_file)
        self.csv_file = pd.concat([_csv_file] * repeat_dataset)
        self.root_dir = root_dir
        self.transform = transforms
        self.normalization = Normalize(mean, std, max_pixel_value)
        self.to_gray = ToGray()

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        
        ribs_img_name = os.path.join(self.root_dir,
                                     self.csv_file.iloc[idx]['ribs'])
        original_image = cv2.imread(ribs_img_name,1)
        clear_img_name = os.path.join(self.root_dir,
                                      self.csv_file.iloc[idx]['cleared'])
        clear_image = cv2.imread(clear_img_name,1)
        
        clear_image = match_histograms(clear_image, original_image)
        clear_image = np.round(clear_image).astype(np.uint8)
        
        train_pair = {'image': original_image, 'target': clear_image}
        if self.transform:
            augmented = self.transform(**train_pair)

        augmented['image'] = self.normalization(image=augmented['image'])['image']
        augmented['image'] = augmented['image'][...,0]
        augmented['target'] = augmented['target'][...,0]
            
        augmented['image'] = np.expand_dims(augmented['image'], axis=0)
        augmented['target'] = np.expand_dims(augmented['target'], axis=0)

        augmented['image'] = torch.from_numpy(augmented['image']).type(torch.float32)
        augmented['target'] = torch.from_numpy(augmented['target']).type(torch.float32)

        sample = {'input': augmented['image'], 
                  'target': augmented['target']}
        return sample
