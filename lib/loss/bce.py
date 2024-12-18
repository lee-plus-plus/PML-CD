import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy


def bce_loss(y_score, y_target, weight=None, reduce='mean'):
    loss = binary_cross_entropy(
        y_score, y_target.float(), weight=weight, reduction='none')

    if reduce == 'mean':
        loss = loss.mean()
    elif reduce == 'sum':
        loss = loss.sum()
    elif reduce == 'none':
        pass
    return loss


def bce_loss_with_logits(y_logit, y_target, *args, **kwargs):
    y_score = torch.sigmoid(y_logit)
    return bce_loss(y_score, y_target, *args, **kwargs)


class BceLoss(nn.Module):
    def __init__(self, reduce='mean'):
        super().__init__()
        self.reduce = reduce

    def forward(self, y_score, y_target, weight=None, reduce=None):
        '''Parameters
        ----------
        y_score: (batch_size, num_classes)
        y_target: (batch_size, num_classes)
        '''
        return bce_loss(
            y_score, y_target, weight=weight,
            reduce=(reduce or self.reduce),
        )


class BceWithLogitsLoss(BceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, y_logit, y_target, *args, **kwargs):
        y_score = torch.sigmoid(y_logit)
        return super().forward(y_score, y_target, *args, **kwargs)
