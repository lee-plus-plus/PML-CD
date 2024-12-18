import numpy as np
from typing import List
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from .utils import to_tensor


@dataclass
class StorableVisualMllDataset:
    image_filenames: List[str]
    labels: np.ndarray
    category_name: List[str]

    def __post_init__(self):
        assert isinstance(self.image_filenames, list)
        assert isinstance(self.labels, np.ndarray)
        assert isinstance(self.category_name, list)

        assert len(self.image_filenames) == self.labels.shape[0]
        assert len(self.category_name) == self.labels.shape[1]

    @property
    def num_samples(self):
        return self.labels.shape[0]

    @property
    def num_classes(self):
        return self.labels.shape[1]


class VisualMllDataset(Dataset):
    def __init__(self, image_filenames, labels, category_name):
        if isinstance(category_name, list):
            category_name = dict(zip(range(labels.shape[1]), category_name))

        self.image_filenames = image_filenames
        self.labels = to_tensor(labels).int()
        self.category_name = category_name

        assert set(self.labels.unique().tolist()) == {0, 1}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        label = self.labels[index, :]

        return image, label

    def __repr__(self):
        return f'VisualMllDataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'

    @property
    def num_samples(self):
        return self.labels.shape[0]

    @property
    def num_classes(self):
        return self.labels.shape[1]
