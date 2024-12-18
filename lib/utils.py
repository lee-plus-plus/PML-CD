import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np
import os
import sys
import warnings
import argparse
import pynvml


def init_cuda_environment(seed=123, device='0'):
    cudnn.benchmark = True
    cudnn.deterministic = True

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = device


def get_model_device(model):
    return next(model.parameters()).device


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def ignore_warnings(func):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


# for argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def best_gpu():
    pynvml.nvmlInit()
    mem_free = [
        pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(i)).free
        for i in range(pynvml.nvmlDeviceGetCount())
    ]
    return mem_free.index(max(mem_free))
