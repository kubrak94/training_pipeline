import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.transform.histogram_matching import match_histograms
from src.train.transforms import Normalize, ToGray
import pywt

def get_decomosition(img):
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    wavelet_decomp = np.zeros((4,)+LL.shape, dtype=np.float32)
    for idx,el in enumerate([LL, LH, HL, HH]):
        wavelet_decomp[idx] = el
    return wavelet_decomp

class WaveletDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, repeat_dataset=1,
                 mean=None, std=None, max_pixel_value=1.0, 
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
        self.normalization = None
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
            
        augmented['image'] = augmented['image'][...,0]
        augmented['target'] = augmented['target'][...,0]        
        
        if not np.isclose(augmented['image'].std(), 0):
            augmented['target'] = (augmented['target'] - augmented['image'].mean())/augmented['image'].std()
            augmented['image'] = (augmented['image'] - augmented['image'].mean())/augmented['image'].std()
        else:
            cv2.imwrite(f'wavelet_{ribs_img_name}.jpg', augmented['image'])
            augmented['target'] = (augmented['target'] - 128.)/64.
            augmented['image'] = (augmented['image'] - 128.)/64.
        
        augmented['image'] = get_decomosition(augmented['image'])
        augmented['target'] = get_decomosition(augmented['target'])

        augmented['image'] = torch.from_numpy(augmented['image']).type(torch.float32)
        augmented['target'] = torch.from_numpy(augmented['target']).type(torch.float32)

        sample = {'input': augmented['image'], 
                  'target': augmented['target']}
        return sample
