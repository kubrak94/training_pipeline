import copy
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ext.histogram_matching import match_histograms


class FPNDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, csv_file, root_dir, multiplier=1., transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.multiplier = multiplier
        self.clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))

    def __len__(self):
        return len(self.csv_file)

    @staticmethod
    def get_brightness(img):
        buf = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brighness = buf[..., 2][buf[..., 2] > 0].mean()
        return brighness

    @staticmethod
    def get_range(img):
        range_ = img[..., 2][img[..., 2] > 0].max() - img[..., 2][img[..., 2] > 0].min()
        return range_
    
    @staticmethod
    def adjust_contrast(img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return final
    
    @staticmethod
    def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def __getitem__(self, idx):
        ribs_mean_brightness, clear_mean_brightness = 147, 150
        ribs_mean_range, clear_mean_range = 223, 222
        
        ribs_img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx]['ribs'])
        original_image = cv2.imread(ribs_img_name)[..., ::-1]
        clear_img_name = os.path.join(self.root_dir,
                                self.csv_file.iloc[idx]['cleared'])
        clear_image = cv2.imread(clear_img_name)[..., ::-1]
        
        #original_image = self.adjust_contrast(original_image)
        #clear_image = self.adjust_contrast(clear_image)
#         original_brightness = self.get_brightness(original_image)
#         original_range = self.get_brightness(original_image)
#         original_image = self.apply_brightness_contrast(original_image,
#                                                         ribs_mean_brightness - original_brightness,
#                                                         ribs_mean_range - original_range
#                                                        )

#         clear_brightness = self.get_brightness(clear_image)
#         clear_range = self.get_brightness(clear_image)
#         clear_image = self.apply_brightness_contrast(clear_image,
#                                                      clear_mean_brightness - clear_brightness,
#                                                      clear_mean_range - clear_range
#                                                     )
                    
        total_image = np.zeros((clear_image.shape[0], clear_image.shape[1], 3), dtype=np.uint8)
        total_image[..., 0] = original_image[..., 0]
        total_image[..., 1:] = clear_image[..., :2]

        if self.transforms:
            #seed = random.randint()
            #random.seed(seed)
            augmented = self.transforms(image=total_image)
            total_image = augmented['image']
            #random.seed(seed)
            #augmented = self.transform(image=total_image)
            #total_image = augmented['image']

        original_image = np.expand_dims(self.clahe.apply(total_image[..., 0]), axis=2)
        clear_image = np.expand_dims(self.clahe.apply(total_image[..., 1]), axis=2)
        
        original_image = np.squeeze(
                         np.stack((original_image, 
                                   original_image, 
                                   original_image), axis=2))
        
        clear_image = np.squeeze(
                         np.stack((clear_image, 
                                   clear_image, 
                                   clear_image), axis=2))

#         original_image = np.squeeze(
#                          np.stack((np.expand_dims(total_image[..., 0], axis=2), 
#                                    np.expand_dims(total_image[..., 0], axis=2), 
#                                    np.expand_dims(total_image[..., 0], axis=2)), axis=2))
        
#         clear_image = np.squeeze(
#                          np.stack((np.expand_dims(total_image[..., 1], axis=2), 
#                                    np.expand_dims(total_image[..., 1], axis=2), 
#                                    np.expand_dims(total_image[..., 1], axis=2)), axis=2))
        
        #clear_image = match_histograms(clear_image, original_image, multichannel=True)
        
#         G = clear_image.copy()
#         gpA = []
#         for i in range(4):
#             G = cv2.pyrDown(G)
#             gpA.append(G)
#         p2, p3, p4, p5 = gpA
        p2 = cv2.resize(clear_image, (clear_image.shape[0] // 2, clear_image.shape[1] // 2))
        p3 = cv2.resize(clear_image, (clear_image.shape[0] // 4, clear_image.shape[1] // 4))
        p4 = cv2.resize(clear_image, (clear_image.shape[0] // 8, clear_image.shape[1] // 8))
        p5 = cv2.resize(clear_image, (clear_image.shape[0] // 16, clear_image.shape[1] // 16))

        original_image = torch.from_numpy(np.moveaxis(original_image, -1, 0).astype(np.float32))

        clear_image = torch.from_numpy(np.moveaxis(clear_image / self.multiplier, -1, 0).astype(np.float32))
        p2 = torch.from_numpy(np.moveaxis(p2 / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        p3 = torch.from_numpy(np.moveaxis(p3 / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        p4 = torch.from_numpy(np.moveaxis(p4 / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
        p5 = torch.from_numpy(np.moveaxis(p5 / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32))

        sample = {'input': original_image, 
                  'target': clear_image, 
                  'p2': p2, 
                  'p3': p3, 
                  'p4': p4, 
                  'p5': p5}
        return sample
