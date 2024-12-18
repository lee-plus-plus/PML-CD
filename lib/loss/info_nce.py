import torch
from torch import nn, Tensor


def self_excluded_log_softmax(logits: Tensor, dim: int = 1) -> Tensor:
    # logits: [batch_size, num_classes]
    # return: [batch_size, num_classes]
    mask = 1 - torch.eye(*logits.shape, device=logits.device)
    logits -= logits.max(dim=dim, keepdim=True).values  # to indicate overflow (too large)
    exp_logits = torch.exp(logits) * mask
    log_proba = logits - torch.log(exp_logits.sum(dim=dim, keepdim=True) + 1e-12)
    return log_proba


def info_nce_loss(query: Tensor, keys: Tensor, positive_mask: Tensor, temperature: float = 0.07,
                  reduce='mean') -> Tensor:
    # query: [num_query, dim_embed]
    # keys: [num_keys, dim_embed]
    # positive_mask: [num_query, num_keys]
    assert query.shape[1] == keys.shape[1], "dim of embedding is not consistant"
    assert query.shape[0] == positive_mask.shape[0], "shape of positive_mask is not aligned"
    assert keys.shape[0] == positive_mask.shape[1], "shape of positive_mask is not aligned"
    assert positive_mask.min() >= 0, "negative value of positive_mask is lack of interpretation"
    # assert 0 not in positive_mask.max(dim=1).values, "no positive pair"

    # loss: [num_query, num_keys]
    logit = query @ keys.t() / temperature
    loss = (-positive_mask * self_excluded_log_softmax(logit, dim=1)).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-12)

    if reduce == 'mean':
        loss = loss.mean()
    elif reduce == 'sum':
        loss = loss.sum()
    elif reduce == 'none':
        pass
    return loss


def sup_con_loss(features: Tensor, labels: Tensor, temperature: float = 0.07, reduce: str = 'mean') -> Tensor:
    # features: [batch_size, dim_embed]
    # labels: [batch_size]
    batch_size, dim_embed = features.shape
    labels = labels.view(batch_size, 1)

    # mask: [batch_size, batch_size]
    # mask[i, j] <=> labels[i] == 1 and labels[j] == 1 and i != j
    mask = (labels.t() * labels) * (1 - torch.eye(batch_size, device=labels.device))

    # loss: [batch_size, batch_size] if reduce == 'none'
    loss = info_nce_loss(features, features, mask, temperature=temperature, reduce=reduce)
    return loss


def multilabel_sup_con_loss(features: Tensor, labels: Tensor, temperature: float = 0.07,
                            reduce: str = 'mean') -> Tensor:
    # features: [batch_size, num_classes, dim_embed]
    # labels: [batch_size, num_classes]
    batch_size, num_classes, dim_embed = features.shape

    loss = torch.stack([
        sup_con_loss(features[:, c, :], labels[:, c], temperature=temperature, reduce='none')
        for c in range(num_classes)
    ], dim=1)  # [batch_size, num_classes, batch_size]

    if reduce == 'mean':
        loss = loss.mean()
    elif reduce == 'sum':
        loss = loss.sum()
    elif reduce == 'none':
        pass
    return loss


# supervised contrastive learning loss
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07, reduce='mean'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduce = reduce

    def forward(self, features, labels, reduce=None):
        # features: [batch_size, dim_embed]
        # labels: [batch_size]
        loss = sup_con_loss(features, labels, temperature=self.temperature,
                            reduce=self.reduce if reduce is None else reduce)
        loss *= (self.temperature / self.base_temperature)
        return loss


# multi-label supervised contrastive learning loss
class MultilabelSupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07, reduce='mean'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduce = reduce

    def forward(self, features, labels, reduce=None):
        # features: [batch_size, num_classes, dim_embed]
        # labels: [batch_size, num_classes]
        loss = multilabel_sup_con_loss(features, labels, temperature=self.temperature,
                                       reduce=self.reduce if reduce is None else reduce)
        loss *= (self.temperature / self.base_temperature)
        return loss
