# import torch
import numpy as np
from glob import glob
# from PIL import Image
from os.path import join, normpath
from .utils import HiddenPrints
from .base import StorableVisualMllDataset, VisualMllDataset
# from torch.utils.data import Dataset


category_name = {
    0: 'airport', 1: 'animal', 2: 'beach', 3: 'bear', 4: 'birds', 5: 'boats', 6: 'book', 7: 'bridge', 8: 'buildings', 9: 'cars',
    10: 'castle', 11: 'cat', 12: 'cityscape', 13: 'clouds', 14: 'computer', 15: 'coral', 16: 'cow', 17: 'dancing', 18: 'dog', 19: 'earthquake',
    20: 'elk', 21: 'fire', 22: 'fish', 23: 'flags', 24: 'flowers', 25: 'food', 26: 'fox', 27: 'frost', 28: 'garden', 29: 'glacier',
    30: 'grass', 31: 'harbor', 32: 'horses', 33: 'house', 34: 'lake', 35: 'leaf', 36: 'map', 37: 'military', 38: 'moon', 39: 'mountain',
    40: 'nighttime', 41: 'ocean', 42: 'person', 43: 'plane', 44: 'plants', 45: 'police', 46: 'protest', 47: 'railroad', 48: 'rainbow', 49: 'reflection',
    50: 'road', 51: 'rocks', 52: 'running', 53: 'sand', 54: 'sign', 55: 'sky', 56: 'snow', 57: 'soccer', 58: 'sports', 59: 'statue',
    60: 'street', 61: 'sun', 62: 'sunset', 63: 'surf', 64: 'swimmers', 65: 'tattoo', 66: 'temple', 67: 'tiger', 68: 'tower', 69: 'town',
    70: 'toy', 71: 'train', 72: 'tree', 73: 'valley', 74: 'vehicle', 75: 'water', 76: 'waterfall', 77: 'wedding', 78: 'whales', 79: 'window',
    80: 'zebra'
}


def load_nuswide_data(
    root_dir, divide, *, cache=True
) -> StorableVisualMllDataset:
    assert divide in {'train', 'test'}

    try:
        image_filenames = np.load(
            join(root_dir, f'image_filenames_{divide}.npy'))
        labels = np.load(join(root_dir, f'labels_{divide}.npy'))

    except FileNotFoundError:
        # print('creating annotation file...')
        with HiddenPrints():
            image_filenames_train_filename = join(
                root_dir, 'ImageList', 'TrainImagelist.txt')
            image_filenames_test_filename = join(
                root_dir, 'ImageList', 'TestImagelist.txt')
            label_train_per_class_filenames = sorted(
                glob(join(root_dir, 'Groundtruth',
                          'TrainTestLabels', 'Labels_*_Train.txt')))
            label_test_per_class_filenames = sorted(
                glob(join(root_dir, 'Groundtruth',
                          'TrainTestLabels', 'Labels_*_Test.txt')))

            if divide == 'train':
                image_filenames_filename = image_filenames_train_filename
                label_per_class_filenames = label_train_per_class_filenames
            elif divide == 'test':
                image_filenames_filename = image_filenames_test_filename
                label_per_class_filenames = label_test_per_class_filenames
            else:
                raise ValueError("divide must in {‘train’, ‘test’}")

            with open(image_filenames_filename, 'r') as file:
                image_filenames = [line.strip().replace('\\', '/')
                                   for line in file.readlines()]
                image_filenames = [join('Flickr', filename)
                                   for filename in image_filenames]

            num_samples = len(image_filenames)
            num_classes = len(category_name)

            labels = np.zeros((num_samples, num_classes), dtype=int)
            for class_index, filename in enumerate(label_per_class_filenames):
                with open(filename, 'r') as file:
                    label_per_class = np.array(
                        [int(line.strip()) for line in file.readlines()],
                        dtype=int)
                    labels[:, class_index] = label_per_class

            if cache:
                np.save(join(root_dir, f'image_filenames_{divide}.npy'),
                        np.array(image_filenames))
                np.save(join(root_dir, f'labels_{divide}.npy'),
                        np.array(labels))

    # to absolute path
    image_filenames = [normpath(join(root_dir, filename))
                       for filename in image_filenames]
    labels = np.array(labels).astype(int)

    return StorableVisualMllDataset(
        image_filenames, labels, list(category_name.values()))


class NusWideDataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_nuswide_data(root_dir, divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'NusWideDataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


r'''
class NusWideDataset(Dataset):
    def __init__(self, root_dir, divide):
        # loading annotations into memory
        with HiddenPrints():
            image_filenames_train_filename = join(
                root_dir, 'ImageList', 'TrainImagelist.txt')
            image_filenames_test_filename = join(
                root_dir, 'ImageList', 'TestImagelist.txt')
            label_train_per_class_filenames = sorted(
                glob(join(root_dir, 'Groundtruth',
                          'TrainTestLabels', 'Labels_*_Train.txt')))
            label_test_per_class_filenames = sorted(
                glob(join(root_dir, 'Groundtruth',
                          'TrainTestLabels', 'Labels_*_Test.txt')))

            if divide == 'train':
                image_filenames_filename = image_filenames_train_filename
                label_per_class_filenames = label_train_per_class_filenames
            elif divide == 'test':
                image_filenames_filename = image_filenames_test_filename
                label_per_class_filenames = label_test_per_class_filenames
            else:
                raise ValueError("divide must in {‘train’, ‘test’}")

            with open(image_filenames_filename, 'r') as file:
                image_filenames = [line.strip().replace('\\', '/')
                                   for line in file.readlines()]
                image_filenames = [join(root_dir, 'Flickr', filename)
                                   for filename in image_filenames]

            # category_name = [re.findall(r'Labels_(.+)_.+\.txt', f_per_cls)[0]
            #                  for f_per_cls in label_per_class_filenames]
            # category_name = dict(enumerate(category_name))

        num_samples = len(image_filenames)
        num_classes = len(category_name)

        labels = torch.zeros((num_samples, num_classes),
                             requires_grad=False, dtype=float)
        for class_index, filename in enumerate(label_per_class_filenames):
            with open(filename, 'r') as file:
                label_per_class = torch.tensor(
                    [int(line.strip()) for line in file.readlines()],
                    dtype=float)
                labels[:, class_index] = label_per_class

        labels = (labels > 0).int()

        self.image_filenames = image_filenames
        self.category_name = category_name
        self.labels = labels
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        label = self.labels[index]

        return image, label

    def image_path(self, index):
        path = self.image_filenames[index]
        return path
'''