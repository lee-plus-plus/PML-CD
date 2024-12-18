from os.path import join, normpath
import numpy as np
# import torch
import pandas as pd
# from PIL import Image
from .base import StorableVisualMllDataset, VisualMllDataset


category_name = {
    0: 'bill_shape:curved_(up_or_down)', 1: 'bill_shape:dagger',
    2: 'bill_shape:hooked', 3: 'bill_shape:needle',
    4: 'bill_shape:hooked_seabird', 5: 'bill_shape:spatulate',
    6: 'bill_shape:all-purpose', 7: 'bill_shape:cone',
    8: 'bill_shape:specialized', 9: 'wing_color:blue',
    10: 'wing_color:brown', 11: 'wing_color:iridescent',
    12: 'wing_color:purple', 13: 'wing_color:rufous',
    14: 'wing_color:grey', 15: 'wing_color:yellow',
    16: 'wing_color:olive', 17: 'wing_color:green',
    18: 'wing_color:pink', 19: 'wing_color:orange',
    20: 'wing_color:black', 21: 'wing_color:white',
    22: 'wing_color:red', 23: 'wing_color:buff',
    24: 'upperparts_color:blue', 25: 'upperparts_color:brown',
    26: 'upperparts_color:iridescent', 27: 'upperparts_color:purple',
    28: 'upperparts_color:rufous', 29: 'upperparts_color:grey',
    30: 'upperparts_color:yellow', 31: 'upperparts_color:olive',
    32: 'upperparts_color:green', 33: 'upperparts_color:pink',
    34: 'upperparts_color:orange', 35: 'upperparts_color:black',
    36: 'upperparts_color:white', 37: 'upperparts_color:red',
    38: 'upperparts_color:buff', 39: 'underparts_color:blue',
    40: 'underparts_color:brown', 41: 'underparts_color:iridescent',
    42: 'underparts_color:purple', 43: 'underparts_color:rufous',
    44: 'underparts_color:grey', 45: 'underparts_color:yellow',
    46: 'underparts_color:olive', 47: 'underparts_color:green',
    48: 'underparts_color:pink', 49: 'underparts_color:orange',
    50: 'underparts_color:black', 51: 'underparts_color:white',
    52: 'underparts_color:red', 53: 'underparts_color:buff',
    54: 'breast_pattern:solid', 55: 'breast_pattern:spotted',
    56: 'breast_pattern:striped', 57: 'breast_pattern:multi-colored',
    58: 'back_color:blue', 59: 'back_color:brown',
    60: 'back_color:iridescent', 61: 'back_color:purple',
    62: 'back_color:rufous', 63: 'back_color:grey',
    64: 'back_color:yellow', 65: 'back_color:olive',
    66: 'back_color:green', 67: 'back_color:pink',
    68: 'back_color:orange', 69: 'back_color:black',
    70: 'back_color:white', 71: 'back_color:red',
    72: 'back_color:buff', 73: 'tail_shape:forked_tail',
    74: 'tail_shape:rounded_tail', 75: 'tail_shape:notched_tail',
    76: 'tail_shape:fan-shaped_tail', 77: 'tail_shape:pointed_tail',
    78: 'tail_shape:squared_tail', 79: 'upper_tail_color:blue',
    80: 'upper_tail_color:brown', 81: 'upper_tail_color:iridescent',
    82: 'upper_tail_color:purple', 83: 'upper_tail_color:rufous',
    84: 'upper_tail_color:grey', 85: 'upper_tail_color:yellow',
    86: 'upper_tail_color:olive', 87: 'upper_tail_color:green',
    88: 'upper_tail_color:pink', 89: 'upper_tail_color:orange',
    90: 'upper_tail_color:black', 91: 'upper_tail_color:white',
    92: 'upper_tail_color:red', 93: 'upper_tail_color:buff',
    94: 'head_pattern:spotted', 95: 'head_pattern:malar',
    96: 'head_pattern:crested', 97: 'head_pattern:masked',
    98: 'head_pattern:unique_pattern', 99: 'head_pattern:eyebrow',
    100: 'head_pattern:eyering', 101: 'head_pattern:plain',
    102: 'head_pattern:eyeline', 103: 'head_pattern:striped',
    104: 'head_pattern:capped', 105: 'breast_color:blue',
    106: 'breast_color:brown', 107: 'breast_color:iridescent',
    108: 'breast_color:purple', 109: 'breast_color:rufous',
    110: 'breast_color:grey', 111: 'breast_color:yellow',
    112: 'breast_color:olive', 113: 'breast_color:green',
    114: 'breast_color:pink', 115: 'breast_color:orange',
    116: 'breast_color:black', 117: 'breast_color:white',
    118: 'breast_color:red', 119: 'breast_color:buff',
    120: 'throat_color:blue', 121: 'throat_color:brown',
    122: 'throat_color:iridescent', 123: 'throat_color:purple',
    124: 'throat_color:rufous', 125: 'throat_color:grey',
    126: 'throat_color:yellow', 127: 'throat_color:olive',
    128: 'throat_color:green', 129: 'throat_color:pink',
    130: 'throat_color:orange', 131: 'throat_color:black',
    132: 'throat_color:white', 133: 'throat_color:red',
    134: 'throat_color:buff', 135: 'eye_color:blue',
    136: 'eye_color:brown', 137: 'eye_color:purple',
    138: 'eye_color:rufous', 139: 'eye_color:grey',
    140: 'eye_color:yellow', 141: 'eye_color:olive',
    142: 'eye_color:green', 143: 'eye_color:pink',
    144: 'eye_color:orange', 145: 'eye_color:black',
    146: 'eye_color:white', 147: 'eye_color:red',
    148: 'eye_color:buff', 149: 'bill_length:about_the_same_as_head',
    150: 'bill_length:longer_than_head', 151: 'bill_length:shorter_than_head',
    152: 'forehead_color:blue', 153: 'forehead_color:brown',
    154: 'forehead_color:iridescent', 155: 'forehead_color:purple',
    156: 'forehead_color:rufous', 157: 'forehead_color:grey',
    158: 'forehead_color:yellow', 159: 'forehead_color:olive',
    160: 'forehead_color:green', 161: 'forehead_color:pink',
    162: 'forehead_color:orange', 163: 'forehead_color:black',
    164: 'forehead_color:white', 165: 'forehead_color:red',
    166: 'forehead_color:buff', 167: 'under_tail_color:blue',
    168: 'under_tail_color:brown', 169: 'under_tail_color:iridescent',
    170: 'under_tail_color:purple', 171: 'under_tail_color:rufous',
    172: 'under_tail_color:grey', 173: 'under_tail_color:yellow',
    174: 'under_tail_color:olive', 175: 'under_tail_color:green',
    176: 'under_tail_color:pink', 177: 'under_tail_color:orange',
    178: 'under_tail_color:black', 179: 'under_tail_color:white',
    180: 'under_tail_color:red', 181: 'under_tail_color:buff',
    182: 'nape_color:blue', 183: 'nape_color:brown',
    184: 'nape_color:iridescent', 185: 'nape_color:purple',
    186: 'nape_color:rufous', 187: 'nape_color:grey',
    188: 'nape_color:yellow', 189: 'nape_color:olive',
    190: 'nape_color:green', 191: 'nape_color:pink',
    192: 'nape_color:orange', 193: 'nape_color:black',
    194: 'nape_color:white', 195: 'nape_color:red',
    196: 'nape_color:buff', 197: 'belly_color:blue',
    198: 'belly_color:brown', 199: 'belly_color:iridescent',
    200: 'belly_color:purple', 201: 'belly_color:rufous',
    202: 'belly_color:grey', 203: 'belly_color:yellow',
    204: 'belly_color:olive', 205: 'belly_color:green',
    206: 'belly_color:pink', 207: 'belly_color:orange',
    208: 'belly_color:black', 209: 'belly_color:white',
    210: 'belly_color:red', 211: 'belly_color:buff',
    212: 'wing_shape:rounded-wings', 213: 'wing_shape:pointed-wings',
    214: 'wing_shape:broad-wings', 215: 'wing_shape:tapered-wings',
    216: 'wing_shape:long-wings', 217: 'size:large_(16_-_32_in)',
    218: 'size:small_(5_-_9_in)', 219: 'size:very_large_(32_-_72_in)',
    220: 'size:medium_(9_-_16_in)', 221: 'size:very_small_(3_-_5_in)',
    222: 'shape:upright-perching_water-like', 223: 'shape:chicken-like-marsh',
    224: 'shape:long-legged-like', 225: 'shape:duck-like',
    226: 'shape:owl-like', 227: 'shape:gull-like',
    228: 'shape:hummingbird-like', 229: 'shape:pigeon-like',
    230: 'shape:tree-clinging-like', 231: 'shape:hawk-like',
    232: 'shape:sandpiper-like', 233: 'shape:upland-ground-like',
    234: 'shape:swallow-like', 235: 'shape:perching-like',
    236: 'back_pattern:solid', 237: 'back_pattern:spotted',
    238: 'back_pattern:striped', 239: 'back_pattern:multi-colored',
    240: 'tail_pattern:solid', 241: 'tail_pattern:spotted',
    242: 'tail_pattern:striped', 243: 'tail_pattern:multi-colored',
    244: 'belly_pattern:solid', 245: 'belly_pattern:spotted',
    246: 'belly_pattern:striped', 247: 'belly_pattern:multi-colored',
    248: 'primary_color:blue', 249: 'primary_color:brown',
    250: 'primary_color:iridescent', 251: 'primary_color:purple',
    252: 'primary_color:rufous', 253: 'primary_color:grey',
    254: 'primary_color:yellow', 255: 'primary_color:olive',
    256: 'primary_color:green', 257: 'primary_color:pink',
    258: 'primary_color:orange', 259: 'primary_color:black',
    260: 'primary_color:white', 261: 'primary_color:red',
    262: 'primary_color:buff', 263: 'leg_color:blue',
    264: 'leg_color:brown', 265: 'leg_color:iridescent',
    266: 'leg_color:purple', 267: 'leg_color:rufous',
    268: 'leg_color:grey', 269: 'leg_color:yellow',
    270: 'leg_color:olive', 271: 'leg_color:green',
    272: 'leg_color:pink', 273: 'leg_color:orange',
    274: 'leg_color:black', 275: 'leg_color:white',
    276: 'leg_color:red', 277: 'leg_color:buff',
    278: 'bill_color:blue', 279: 'bill_color:brown',
    280: 'bill_color:iridescent', 281: 'bill_color:purple',
    282: 'bill_color:rufous', 283: 'bill_color:grey',
    284: 'bill_color:yellow', 285: 'bill_color:olive',
    286: 'bill_color:green', 287: 'bill_color:pink',
    288: 'bill_color:orange', 289: 'bill_color:black',
    290: 'bill_color:white', 291: 'bill_color:red',
    292: 'bill_color:buff', 293: 'crown_color:blue',
    294: 'crown_color:brown', 295: 'crown_color:iridescent',
    296: 'crown_color:purple', 297: 'crown_color:rufous',
    298: 'crown_color:grey', 299: 'crown_color:yellow',
    300: 'crown_color:olive', 301: 'crown_color:green',
    302: 'crown_color:pink', 303: 'crown_color:orange',
    304: 'crown_color:black', 305: 'crown_color:white',
    306: 'crown_color:red', 307: 'crown_color:buff',
    308: 'wing_pattern:solid', 309: 'wing_pattern:spotted',
    310: 'wing_pattern:striped', 311: 'wing_pattern:multi-colored',
}


def load_cub200_data(
    root_dir, divide, *, cache=True
) -> StorableVisualMllDataset:
    assert divide in ['train', 'test', 'undivided']

    try:
        image_filenames = np.load(
            join(root_dir, f'image_filenames_{divide}.npy'))
        labels = np.load(join(root_dir, f'labels_{divide}.npy'))

    except FileNotFoundError:
        # print('creating annotation file...')

        image_filenames = np.genfromtxt(
            join(root_dir, 'images.txt'), dtype=str)[:, 1]
        is_train = np.genfromtxt(
            join(root_dir, 'train_test_split.txt'), dtype=int)[:, 1] == 1

        targets = pd.read_csv(
            join(root_dir, 'attributes', 'image_attribute_labels.txt'),
            delimiter=' ',
            header=None,
            names=['image_id', 'attribute_id',
                   'is_present', 'certainty_id', 'time'],
            usecols=['image_id', 'attribute_id', 'is_present']
        ).to_numpy()

        labels = np.zeros((len(image_filenames), len(category_name)),
                          dtype=int)
        labels[targets[:, 0] - 1, targets[:, 1] - 1] = targets[:, 2]

        image_filenames = [join('images', filename)
                           for filename in image_filenames]

        if divide == 'train':
            mask = is_train
        elif divide == 'test':
            mask = ~is_train
        else:
            mask = np.ones_like(is_train)

        image_filenames = np.array(image_filenames)[mask]
        labels = labels[mask, :]

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


class Cub200Dataset(VisualMllDataset):
    def __init__(self, root_dir, divide):
        data = load_cub200_data(root_dir, divide)
        super().__init__(data.image_filenames, data.labels, data.category_name)

    def __repr__(self):
        return f'Cub200Dataset(num_samples={self.num_samples}, ' \
               f'num_classes={self.num_classes})'


'''
class Cub200Dataset(torch.utils.data.Dataset):
    def __init__(self, root, divide):
        assert divide in ['train', 'test', 'undivided']

        image_filenames = np.genfromtxt(
            join(root, 'images.txt'), dtype=str)[:, 1]
        is_train = np.genfromtxt(
            join(root, 'train_test_split.txt'), dtype=int)[:, 1] == 1

        targets = pd.read_csv(
            join(root, 'attributes', 'image_attribute_labels.txt'),
            delimiter=' ',
            header=None,
            names=['image_id', 'attribute_id',
                   'is_present', 'certainty_id', 'time'],
            usecols=['image_id', 'attribute_id', 'is_present']
        ).to_numpy()

        labels = np.zeros((len(image_filenames), len(category_name)),
                          dtype=int)
        labels[targets[:, 0] - 1, targets[:, 1] - 1] = targets[:, 2]

        image_filenames = [join(root, 'images', filename)
                           for filename in image_filenames]

        if divide == 'train':
            mask = is_train
        elif divide == 'test':
            mask = ~is_train
        else:
            mask = np.ones_like(is_train)

        image_filenames = np.array(image_filenames)[mask]
        labels = labels[mask, :]

        num_samples = len(image_filenames)
        num_classes = len(category_name)

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
'''
