from albumentations.core.composition import *
from albumentations.augmentations.transforms import *
from albumentations.pytorch import ToTensor
from albumentations.core.transforms_interface import ImageOnlyTransform
import cv2


class ToGray(ImageOnlyTransform):
    """Convert the input RGB image to grayscale.
    Args:
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image
    Image types:
        uint8, float32
    """

    def apply(self, img, **params):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
