import torch
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset
from .utils import add_partial_noise, add_multilabel_noise, get_split_indices
from typing import Type, List, Tuple, Optional
import abc


# abstract class
class DownsampledDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class IndexedDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class PartiallyNoisyDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class MultilabelNoisyDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class TransformedDataset(metaclass=abc.ABCMeta):
    pass


# abstract class
class GetItemsDataset(metaclass=abc.ABCMeta):
    pass


def downsampled_dataset_class(
    dataset_class: Type[Dataset],
    downsample_ratio: float,
) -> Type[Dataset]:
    '''let the dataset iterate with index
    '''
    class _DownsampledDataset(dataset_class, DownsampledDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            indices = get_split_indices(self.num_samples, [downsample_ratio], shuffle=True)[0]
            self.image_filenames = [self.image_filenames[i] for i in indices]
            self.labels = self.labels[indices]

    return _DownsampledDataset


def indexed_dataset_class(
    dataset_class: Type[Dataset],
) -> Type[Dataset]:
    '''let the dataset iterate with index
    '''
    class _IndexedDataset(dataset_class, IndexedDataset):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result + (index,)

    return _IndexedDataset


def partially_noisy_dataset_class(
    dataset_class: Type[Dataset],
    noise_rate: float,
) -> Type[Dataset]:
    '''let the dataset iterate with partial labels
    '''
    class _PartiallyNoisyDataset(dataset_class, PartiallyNoisyDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)
            self.partial_labels = add_partial_noise(self.labels, noise_rate)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result + (self.partial_labels[index],)

    return _PartiallyNoisyDataset


def multilabel_noisy_dataset_class(
    dataset_class: Type[Dataset],
    pos_noise_rate: float, neg_noise_rate: float,
) -> Type[Dataset]:
    '''let the dataset iterate with partial labels
    '''
    class _MultilabelNoisyDataset(dataset_class, MultilabelNoisyDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)
            self.noisy_labels = add_multilabel_noise(
                self.labels, pos_noise_rate, neg_noise_rate)

        def __getitem__(self, index):
            result = super().__getitem__(index)
            return result + (self.noisy_labels[index],)

    return _MultilabelNoisyDataset


def transformed_dataset_class(
    dataset_class: Type[Dataset],
    transforms: List[Compose],
    flatten: bool,
) -> Type[Dataset]:
    '''let the dataset iterate with transformed images
    '''
    class _TransformedDataset(dataset_class, TransformedDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)
            self.transforms = transforms
            self._transformed_flatten = flatten

        def __getitem__(self, index):
            result = super().__getitem__(index)
            image, else_item = result[0], result[1:]  # (image, labels)
            images = tuple([t(image) for t in self.transforms])

            # if flatten: (transformed_img_1, transformed_img_2, ..., labels)
            # else:       ((transformed_img_1, transformed_img_2, ...), labels)
            images = images if self._transformed_flatten else (images,)
            return images + else_item

    return _TransformedDataset


def getitems_dataset_class(dataset_class: Type[Dataset]) -> Type[Dataset]:
    '''let the dataset support __getitems__
    examples:

    >>> from torch.utils.data import DataLoader
    >>> dataset = TableMllDataset(features, labels)
    >>> loader = DataLoader(dataset, batch_size=64)

    by supporting dataset.__getitems__, batch loading is much faster

    >>> from torch.utils.data import DataLoader
    >>> dataset = getitems_dataset_class(TableMllDataset)(features, labels)
    >>> loader = DataLoader(dataset, batch_size=64, collate_fn=lambda x: x)
    '''
    class _GetItemsDataset(dataset_class, GetItemsDataset):
        def __init__(self, *args, **kwargs):
            dataset_class.__init__(self, *args, **kwargs)

        def __getitems__(self, indices):
            return self.__getitem__(indices)

    return _GetItemsDataset


def customize(
    dataset_class: Type[Dataset], *,
    downsample_ratio: float = 1.0,
    add_index: bool = False,
    add_partial_noise: bool = False,
    add_multilabel_noise: bool = False,
    noise_rate: float = 0.0,
    pos_noise_rate: float = 0.0,
    neg_noise_rate: float = 0.0,
    transforms: List[Compose] = [Compose([Resize((224, 224)), ToTensor()])],
    flatten: bool = True,
    add_getitems: bool = False,
):
    if downsample_ratio != 1.0:
        dataset_class = downsampled_dataset_class(dataset_class, downsample_ratio)
    if transforms:
        dataset_class = transformed_dataset_class(
            dataset_class, transforms, flatten)
    if add_partial_noise:
        dataset_class = partially_noisy_dataset_class(
            dataset_class, noise_rate)
    if add_multilabel_noise:
        dataset_class = multilabel_noisy_dataset_class(
            dataset_class, pos_noise_rate, neg_noise_rate)
    if add_index:
        dataset_class = indexed_dataset_class(dataset_class)
    if add_getitems:
        dataset_class = getitems_dataset_class(dataset_class)

    return dataset_class
