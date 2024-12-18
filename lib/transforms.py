import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch import Tensor
from PIL import Image, ImageDraw
import random
import numpy as np


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]


class Normalize(transforms.Normalize):
    def __init__(self, mean=imagenet_mean, std=imagenet_std, inplace=False):
        super().__init__(mean, std, inplace)
        pass
        

class UnNormalize(torch.nn.Module):
    """Un-normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will un-normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = input[channel] * std[channel] + mean[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """
    def __init__(self, mean=imagenet_mean, std=imagenet_std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = list(-torch.tensor(self.mean) / torch.tensor(self.std))
        std = list(1 / torch.tensor(self.std))
        # std' = 1 / std, mean' = -mean / std
        return F.normalize(tensor, mean, std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomCutout(torch.nn.Module):
    """Cutout the given image with a random color rectangle mask

    Args:
        cutout_factor(float, betweeen 0 and 1): Control the size of mask. 
        inplace(bool,optional): Bool to make this operation in-place.

    """
    def __init__(self, cutout_factor=0.5, inplace=False):
        super().__init__()
        self.cutout_factor = cutout_factor
        self.inplace = inplace

    def forward(self, image: Image.Image):
        if self.inplace == False:
            image = image.copy()

        img_draw = ImageDraw.Draw(image)
        h, w = image.size
        h_cutout = int(self.cutout_factor * h)
        w_cutout = int(self.cutout_factor * w)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return image