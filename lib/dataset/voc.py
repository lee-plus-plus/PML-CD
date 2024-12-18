# import torch
import numpy as np
from os.path import join, normpath, relpath
from .utils import HiddenPrints
from .base import StorableVisualMllDataset, VisualMllDataset
import torchvision
# from torch.utils.data import Dataset


category_map = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
    'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

category_name = dict(zip(category_map.values(), category_map.keys()))


def load_voc_data(
    root_dir, year, divide, *, cache=True
) -> StorableVisualMllDataset:
    assert year in {'2007', '2012'}
    assert divide in {'train', 'val', 'trainval', 'test'}
    assert (year, divide) != ('2012', 'test')

    try:
        image_filenames = np.load(
            join(root_dir, f'image_filenames_{divide}{year}.npy'))
        labels = np.load(join(root_dir, f'labels_{divide}{year}.npy'))

    except FileNotFoundError:
        # loading annotations into memory
        with HiddenPrints():
            voc = torchvision.datasets.VOCDetection(
                root=root_dir, year=year,
                image_set=divide, download=False
            )

        # creating multi-label annotations
        annotations = [tgt['annotation']['object'] for img, tgt in voc]
        categories = [sorted(list({str(seg_box['name'])
                                   for seg_box in elem}))
                      for elem in annotations]
        targets = [[category_map[category]
                    for category in elem] for elem in categories]

        num_samples, num_classes = len(voc), len(category_map)
        labels = np.zeros((num_samples, num_classes), dtype=int)
        for row in range(num_samples):
            labels[row, targets[row]] = 1

        image_filenames = [relpath(filename, root_dir)
                           for filename in voc.images]

        if cache:
            np.save(join(root_dir, f'image_filenames_{divide}{year}.npy'),
                    np.array(image_filenames))
            np.save(join(root_dir, f'labels_{divide}{year}.npy'),
                    np.array(labels))

    image_filenames = [normpath(join(root_dir, filename))
                       for filename in image_filenames]
    labels = np.array(labels).astype(int)

    return StorableVisualMllDataset(
        image_filenames, labels, list(category_name.values()))


class Voc2007Dataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_voc_data(root_dir, '2007', divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'Voc2007Dataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


class Voc2012Dataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_voc_data(root_dir, '2012', divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'Voc2012Dataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


'''
class VocDataset(Dataset):
    def __init__(self, root_dir, year, divide):
        assert year in {'2007', '2012'}
        assert divide in {'train', 'val', 'trainval', 'test'}
        assert (year, divide) != ('2012', 'test')

        # loading annotations into memory
        with HiddenPrints():
            voc = torchvision.datasets.VOCDetection(
                root=root_dir, year=year,
                image_set=divide, download=False
            )

        # creating multi-label annotations
        annotations = [tgt['annotation']['object'] for img, tgt in voc]
        categories = [sorted(list({str(seg_box['name'])
                                   for seg_box in elem}))
                      for elem in annotations]
        targets = [[category_map[category]
                    for category in elem] for elem in categories]

        num_samples, num_classes = len(voc), len(category_map)
        labels = torch.zeros((num_samples, num_classes),
                             requires_grad=False, dtype=float)
        for row in range(num_samples):
            labels[row, targets[row]] = 1
        labels = (labels > 0).int()

        self.voc = voc
        self.image_filenames = self.voc.images
        self.category_name = category_name
        self.labels = labels
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, index):
        image, _ = self.voc[index]
        label = self.labels[index, :]

        return image, label

'''
