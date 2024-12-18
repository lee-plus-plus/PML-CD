import numpy as np
# import torch
import torchvision
import pycocotools
# from PIL import Image
from os.path import join, normpath
from .utils import HiddenPrints
# from torch.utils.data import Dataset
from .base import StorableVisualMllDataset, VisualMllDataset

category_map = {
    '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9,
    '11': 10, '13': 11, '14': 12, '15': 13, '16': 14, '17': 15, '18': 16, '19': 17, '20': 18, '21': 19,
    '22': 20, '23': 21, '24': 22, '25': 23, '27': 24, '28': 25, '31': 26, '32': 27, '33': 28, '34': 29,
    '35': 30, '36': 31, '37': 32, '38': 33, '39': 34, '40': 35, '41': 36, '42': 37, '43': 38, '44': 39,
    '46': 40, '47': 41, '48': 42, '49': 43, '50': 44, '51': 45, '52': 46, '53': 47, '54': 48, '55': 49,
    '56': 50, '57': 51, '58': 52, '59': 53, '60': 54, '61': 55, '62': 56, '63': 57, '64': 58, '65': 59,
    '67': 60, '70': 61, '72': 62, '73': 63, '74': 64, '75': 65, '76': 66, '77': 67, '78': 68, '79': 69,
    '80': 70, '81': 71, '82': 72, '84': 73, '85': 74, '86': 75, '87': 76, '88': 77, '89': 78, '90': 79
}

category_name = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
    15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "sofa", 58: "pottedplant", 59: "bed",
    60: "diningtable", 61: "toilet", 62: "tvmonitor", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
    70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
    75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}


def load_coco_data(
    root_dir, year, divide, *, cache=True
) -> StorableVisualMllDataset:

    assert year in {'2014', '2017'}
    assert divide in {'train', 'val'}

    try:
        image_filenames = np.load(
            join(root_dir, f'image_filenames_{divide}{year}.npy'))
        labels = np.load(join(root_dir, f'labels_{divide}{year}.npy'))

    except FileNotFoundError:
        # print('creating annotation file...')

        # loading annotations into memory
        with HiddenPrints():
            coco = torchvision.datasets.CocoDetection(
                root=join(root_dir, f'{divide}{year}/'),
                annFile=join(root_dir,
                             f'annotations/instances_{divide}{year}.json')
            )

        # creating multi-label annotations
        categories = [coco._load_target(coco.ids[i])
                      for i in range(len(coco))]
        categories = [sorted(list({str(seg_box['category_id'])
                                   for seg_box in elem}))
                      for elem in categories]

        # map category to class-index
        targets = [[category_map[category]
                    for category in elem] for elem in categories]

        # generate label matrix
        num_samples, num_classes = len(targets), len(category_map)
        labels = np.zeros((num_samples, num_classes), dtype=int)
        for row in range(num_samples):
            labels[row, targets[row]] = 1

        # generate image filename (relative path)
        image_filenames = [
            join(f'{divide}{year}/', elem['file_name'])
            for elem in coco.coco.loadImgs(coco.ids)]

        if cache:
            np.save(join(root_dir, f'image_filenames_{divide}{year}.npy'),
                    np.array(image_filenames))
            np.save(join(root_dir, f'labels_{divide}{year}.npy'),
                    np.array(labels))

    # to absolute path
    image_filenames = [normpath(join(root_dir, filename))
                       for filename in image_filenames]
    labels = np.array(labels).astype(int)

    return StorableVisualMllDataset(
        image_filenames, labels, list(category_name.values()))


class Coco2014Dataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_coco_data(root_dir, '2014', divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'Coco2014Dataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


class Coco2017Dataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_coco_data(root_dir, '2017', divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'Coco2017Dataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


'''
class CocoDataset(Dataset):
    def __init__(self, root_dir, year, divide):
        assert year in {'2014', '2017'}
        assert divide in {'train', 'val'}

        try:
            image_filenames = np.load(
                join(root_dir, f'image_filenames_{divide}.npy'))
            labels = np.load(join(root_dir, f'labels_{divide}.npy'))

        except FileNotFoundError:
            print('creating annotation file...')

            # loading annotations into memory
            with HiddenPrints():
                coco = torchvision.datasets.CocoDetection(
                    root=join(root_dir, f'{divide}{year}/'),
                    annFile=join(
                        root_dir, f'annotations/instances_{divide}{year}.json')
                )

            # creating multi-label annotations
            categories = [coco._load_target(coco.ids[i])
                          for i in range(len(coco))]
            categories = [sorted(list({str(seg_box['category_id'])
                                       for seg_box in elem}))
                          for elem in categories]
            # map category to class-index
            targets = [[category_map[category]
                        for category in elem] for elem in categories]
            # generate label matrix
            num_samples, num_classes = len(targets), len(category_map)
            labels = np.zeros((num_samples, num_classes), dtype=int)
            for row in range(num_samples):
                labels[row, targets[row]] = 1

            image_filenames = [
                join(f'{divide}{year}/', elem['file_name'])
                for elem in self.coco.coco.loadImgs(self.coco.ids)]

            np.save(join(root_dir, f'image_filenames_{divide}.npy'),
                    np.array(image_filenames))
            np.save(join(root_dir, f'labels_{divide}.npy'), np.array(labels))

        finally:
            image_filenames = [join(root_dir, filename)
                               for filename in image_filenames]
            labels = torch.from_numpy(labels)
            assert len(image_filenames) == len(labels)

        self.category_name = category_name
        self.image_filenames = image_filenames
        self.labels = torch.from_numpy(labels)
        self.num_samples = num_samples
        self.num_classes = num_classes

    @property
    def num_classes(self):
        return self.labels.shape[0]

    @property
    def num_samples(self):
        return self.labels.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('RGB')
        label = self.labels[index, :]

        return image, label
'''
