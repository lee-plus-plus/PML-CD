import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def random_rgb_image(image_size=(224, 224)):
    shape = image_size + (3,)
    arr = np.random.randint(0, 256, shape)
    img = Image.fromarray(arr.astype('uint8')).convert('RGB')
    return img


class RandDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=20, image_size=(224, 224)):
        # generate random images, labels
        images = [random_rgb_image(image_size) for i in range(num_samples)]
        labels = torch.randint(0, 2, (num_samples, num_classes)).int()
        category_name = {i: str(i) for i in range(num_classes)}

        self.category_name = category_name
        self.images = images
        self.labels = labels
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index, :]

        return image, label

    def __repr__(self):
        return f'RandDataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'
