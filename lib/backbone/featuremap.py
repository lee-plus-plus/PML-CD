import torch
from torch import nn
from .cnn import cnn_backbone_info, build_cnn_backbone


class IntermediateLayerExtracter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layer: str) -> None:
        if return_layer not in [name for name, _ in model.named_children()]:
            raise ValueError("return_layer are not present in model")
        layers = dict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)
        self.return_layer = return_layer

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        return x


def to_featuremap_backbone(cnn_backbone):
    layer_name = cnn_backbone_info(cnn_backbone)['layer_name_featuremap']
    featuremap_backbone = IntermediateLayerExtracter(cnn_backbone, layer_name)
    return featuremap_backbone


def build_featuremap_backbone(name, *, pretrained=False):
    cnn_backbone = build_cnn_backbone(name, pretrained=pretrained)
    featuremap_backbone = to_featuremap_backbone(cnn_backbone)
    return featuremap_backbone
