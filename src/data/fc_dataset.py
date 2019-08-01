import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.train.transforms import Normalize, ToGray


class FCDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, repeat_dataset=1,
                 mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25), max_pixel_value=1.0, 
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
        original_image = cv2.imread(ribs_img_name)[..., ::-1]
        clear_img_name = os.path.join(self.root_dir,
                                      self.csv_file.iloc[idx]['cleared'])
        clear_image = cv2.imread(clear_img_name)[..., ::-1]

        if self.transform:
            augmented = self.transform(image=original_image, image2=clear_image)
            original_image = self.to_gray(image=augmented['image'])['image']
            original_image = self.normalization(image=original_image)['image']
            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            clear_image = cv2.cvtColor(augmented['image2'], cv2.COLOR_RGB2GRAY)

        original_image = np.expand_dims(original_image, axis=2)
        clear_image = np.expand_dims(clear_image, axis=2)
        
        original_image = np.moveaxis((original_image), -1, 0)
        clear_image = np.moveaxis((clear_image), -1, 0)

        original_image = torch.from_numpy(original_image).type(torch.float32)
        clear_image = torch.from_numpy(clear_image).type(torch.float32)

        sample = {'input': original_image, 
                  'target': clear_image}
        return sample
