import torch
from torch import nn


class GroupLinear(nn.Module):

    def __init__(self, num_classes, in_features, out_features, *, bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_features, out_features))
        torch.nn.init.xavier_normal_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes, out_features))
            torch.nn.init.constant_(self.bias, 0)
        else:
            self.bias = None

    def forward(self, x):
        batch_size, num_classes, in_features = x.shape
        y = torch.einsum('bci,cij->bcj', x, self.weight)

        if self.bias is not None:
            y += self.bias[None, :, :]

        return y
