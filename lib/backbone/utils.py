import torch
import torchvision
import timm
from functools import reduce


def get_layer_by_name(model, name):
    return reduce(
        lambda layer, name: getattr(layer, name), 
        [model] + 'name'.split('.')
    )


def set_layer_by_name(model, name, value):
    name_list = name.split('.')
    parent = reduce(
        lambda layer, name: getattr(layer, name), 
        [model] + name_list[:-1]
    )
    setattr(parent, name_list[-1], value)