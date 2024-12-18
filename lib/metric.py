import torch
from torch import Tensor
from typing_extensions import Optional
from torchmetrics import Metric
from torchmetrics.classification import (
    MultilabelHammingDistance,
    MultilabelRankingLoss,
    MultilabelAveragePrecision,
    MultilabelAccuracy,
    MultilabelF1Score,
)
from torchmetrics.functional.classification import (
    multilabel_hamming_distance,
    multilabel_ranking_loss,
    multilabel_average_precision,
    average_precision,
)


class MultilabelAveragePrecision2(Metric):
    '''
    sample-wise Multi-label Average Precision, 
    equals to `sklearn.metrics.average_precision_score`
    '''
    is_differentiable = None
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=torch.zeros(
            (0, num_labels), dtype=float), dist_reduce_fx="cat")
        self.add_state("target", default=torch.zeros(
            (0, num_labels), dtype=int), dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds = torch.cat([self.preds, preds])
        self.target = torch.cat([self.target, target])

    def compute(self):
        ap_score = torch.tensor([
            average_precision(preds, target, task='binary')
            for preds, target in zip(self.preds, self.target)
        ])
        ap_score = ap_score[~ap_score.isnan()]
        return ap_score.mean()
