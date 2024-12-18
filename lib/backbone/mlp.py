import torch
from torch import nn


class MlpNet(nn.Module):
    # basic MLP network
    def __init__(self, in_features, out_features, hidden_layer_sizes=[]):
        super(MlpNet, self).__init__()
        sizes = [in_features] + list(hidden_layer_sizes) + [out_features]

        sequential = []
        sequential.append(torch.nn.Linear(sizes[0], sizes[1], bias=True))
        for in_, out_ in zip(sizes[1:-1], sizes[2:]):
            sequential.append(torch.nn.ReLU(inplace=True))
            sequential.append(torch.nn.Linear(in_, out_, bias=True))
        # sequential.append(nn.functional.sigmoid())

        self.sequential = torch.nn.Sequential(*sequential)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.sequential(x)
        return x

