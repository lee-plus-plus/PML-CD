import numpy as np
import torch
import os
import sys
from torch.utils.data import Dataset, SubsetRandomSampler
from typing import List, Union, Optional


def to_tensor(
    X: Union[np.ndarray, torch.Tensor],
    copy: bool = False
) -> torch.Tensor:
    '''transform np.ndarry / torch.Tensor to torch.Tensor
    '''
    if isinstance(X, np.ndarray):
        return torch.from_numpy(X)
    if isinstance(X, torch.Tensor):
        return X.clone() if copy else X
    else:
        raise ValueError(f'cannot convert {type(X)} to torch.Tensor')


def get_split_samplers(
    dataset: Dataset,
    split_ratio: List[float] = [0.5, 0.5],
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> List[SubsetRandomSampler]:
    '''split the dataset, return corresponding subset samplers
    '''
    indices = list(range(len(dataset)))
    split_ratio = (np.cumsum([0] + split_ratio) * len(dataset)).astype(int)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    splited_indices = [indices[st: ed]
                       for st, ed in zip(split_ratio[:-1], split_ratio[1:])]
    splited_sampler = [SubsetRandomSampler(idxs) for idxs in splited_indices]

    return splited_sampler


class HiddenPrints:
    '''supress the stdout / stderr during the context
    '''

    def __enter__(self, stdout=True, stderr=False):
        if stdout:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        if stderr:
            self._original_stdout = sys.stderr
            sys.stderr = open(os.devnull, 'w')
        self._hide_stdout = stdout
        self._hide_stderr = stderr

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hide_stdout:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self._hide_stderr:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def add_partial_noise(
    labels: torch.Tensor,
    noise_rate: float,
) -> torch.Tensor:
    '''add partial noise into labels with uniform probability
    '''
    noise = torch.rand(labels.shape) < noise_rate
    partial_labels = (labels.int() | noise.int()).int()
    return partial_labels


def add_multilabel_noise(
    labels: torch.Tensor,
    pos_noise_rate: float, neg_noise_rate: float,
) -> torch.Tensor:
    '''add partial noise into labels with uniform probability
    '''
    pos_noise = torch.rand(labels.shape) < pos_noise_rate
    neg_noise = torch.rand(labels.shape) < neg_noise_rate
    noise = ((labels == 1) & pos_noise) | ((labels == 0) & neg_noise)
    noisy_labels = (labels.int() ^ noise.int()).int()
    return noisy_labels


def get_split_indices(num_samples: int, split_ratio: List[float], shuffle=True):
    indices = list(range(num_samples))
    split_ratio = (np.cumsum([0] + split_ratio) * num_samples).astype(int)

    if shuffle:
        np.random.shuffle(indices)

    splited_indices = [
        indices[st: ed]
        for st, ed in zip(split_ratio[:-1], split_ratio[1:])
    ]
    return splited_indices
