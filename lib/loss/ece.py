import torch


def ece_loss(y_score, y_target, n_bins=20, reduce='mean'):
    bins = torch.linspace(0, 1, n_bins + 1)
    x = (bins[1:] + bins[:-1]) / 2
    hist_pos = torch.histogram(y_score[y_target == 1], bins=bins).hist / y_score.numel()
    hist_neg = torch.histogram(y_score[y_target == 0], bins=bins).hist / y_score.numel()
    hist = hist_pos + hist_neg + 1e-4
    ratio = hist_pos / hist
    error = torch.abs(ratio - x)
    return (hist * error).sum()
