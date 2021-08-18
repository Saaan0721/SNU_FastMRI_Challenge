import numpy as np
import torch
from albumentations import (VerticalFlip, HorizontalFlip, Flip, RandomRotate90, Rotate, ShiftScaleRotate, CenterCrop, OpticalDistortion, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
                            RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CLAHE, ChannelShuffle, InvertImg, RandomGamma, ToGray, PadIfNeeded 
                           )

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)


def hori_filp_image(image):
    aug = HorizontalFlip(p=1)
    image = aug.apply(image)
    return image


def vert_filp_image(image):
    aug = VerticalFlip(p=1)
    image = aug.apply(image)
    return image


class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = to_tensor(input)
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input, target, maximum, fname, slice


class DataHoriFlip:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = hori_filp_image(input)
        input = to_tensor(input)
        if not self.isforward:
            target = hori_filp_image(target)
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input, target, maximum, fname, slice


class DataVertFlip:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, input, target, attrs, fname, slice):
        input = vert_filp_image(input)
        input = to_tensor(input)
        if not self.isforward:
            target = vert_filp_image(target)
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        return input, target, maximum, fname, slice
