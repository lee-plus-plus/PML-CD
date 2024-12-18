from glob import glob
from lib.loss import mse_loss, bce_loss
from torch import nn
import argparse
import torch


class MyWeighter(nn.Module):

    def __init__(self, bins):
        super().__init__()
        self.bins = bins
        self.body = nn.Sequential(
            nn.Linear(bins, bins),
            nn.LeakyReLU(inplace=True),
            nn.Linear(bins, bins),
        )

    def forward(self, y_score, y_partial):
        batch_size, num_classes = y_score.shape
        device = next(self.parameters()).device

        bins = torch.linspace(0.0, 1.0, self.bins + 1)
        x = (bins[1:] + bins[:-1]) / 2
        weight = torch.ones_like(y_score).float()

        for c in range(num_classes):
            mask = y_partial[:, c] == 1
            score = y_score[:, c][mask]
            hist = torch.histogram(score.cpu(), bins=bins).hist / score.numel()
            y = self.fit(hist.to(device))

            x_ = torch.cat([torch.tensor([0], device=x.device), x, torch.tensor([1], device=x.device)]).to(device)
            y_ = torch.cat([torch.tensor([0], device=y.device), y, torch.tensor([1], device=y.device)])
            w = self.interpolate(score, x=x_, y=y_)

            weight[mask, c] = w
        return weight

    def forward_optimal(self, y_score, y_partial, y_true):
        batch_size, num_classes = y_score.shape
        device = next(self.parameters()).device

        bins = torch.linspace(0.0, 1.0, self.bins + 1)
        x = (bins[1:] + bins[:-1]) / 2
        weight = torch.ones_like(y_score).float()

        for c in range(num_classes):
            mask = y_partial[:, c] == 1
            tp_mask = (y_partial[:, c] == 1) & (y_true[:, c] == 1)
            fp_mask = (y_partial[:, c] == 1) & (y_true[:, c] == 0)
            score = y_score[:, c][mask]
            tp_score = y_score[:, c][tp_mask]
            fp_score = y_score[:, c][fp_mask]
            tp_hist = torch.histogram(tp_score.cpu(), bins=bins).hist / (tp_score.numel() + fp_score.numel())
            fp_hist = torch.histogram(fp_score.cpu(), bins=bins).hist / (tp_score.numel() + fp_score.numel())
            y = tp_hist / (tp_hist + fp_hist + 1e-4)
            y = torch.cat([torch.tensor([0], device=y.device), (y[2:] + y[1:-1] + y[:-2]) / 3,
                torch.tensor([1], device=y.device)]).cummax(dim=-1).values

            x_ = torch.cat([torch.tensor([0], device=x.device), x, torch.tensor([1], device=x.device)]).to(device)
            y_ = torch.cat([torch.tensor([0], device=y.device), y, torch.tensor([1], device=y.device)])
            w = self.interpolate(score, x=x_, y=y_)

            weight[mask, c] = w
        return weight

    def fit(self, hist):
        hist = torch.logit(hist, eps=1e-06)
        y_delta = self.body(hist)
        y = torch.cumsum(torch.softmax(y_delta, dim=-1), dim=-1)
        return y

    def interpolate(self, query, x, y):
        query_flatten = query.view(-1)
        indices = torch.searchsorted(x, query_flatten) - 1
        indices = torch.clamp(indices, 0, len(x) - 2)

        x0 = x[indices]
        x1 = x[indices + 1]
        y0 = y[indices]
        y1 = y[indices + 1]

        slope = (y1 - y0) / (x1 - x0)
        result = y0 + slope * (query_flatten - x0)
        return result.view(query.shape)



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Calibrator training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add = parser.add_argument
    parser.add('src', type=str)
    parser.add('target', type=str)
    parser.add('--name', default='', type=str)
    parser.add('--n-bins', default=20, type=int)
    parser.add('--score-name', default='Y_score_clf', type=str, choices=['Y_score_clf', 'Y_score_proto', 'Y_score_fus'])

    args = parser.parse_args(args)
    return args


def get_ckpt_hist(checkpoint, epoch, class_idx, score_name='Y_score_clf', bins=20):
    labels = checkpoint[1]['labels']
    partial_labels = checkpoint[1]['partial_labels']

    masks = {name: ((partial_labels == p) & (labels == t))
             for (name, p, t) in [('TP', 1, 1), ('FP', 1, 0)]}

    Y_score = checkpoint[epoch][score_name][:, class_idx]
    tp, fp = masks['TP'][:, class_idx], masks['FP'][:, class_idx]
    p_count = torch.histogram(Y_score[tp | fp], bins=bins).hist / Y_score[tp | fp].numel()
    tp_count = torch.histogram(Y_score[tp], bins=bins).hist / Y_score[tp | fp].numel()
    target = tp_count / (1e-4 + p_count)
    target_smoothed = torch.cat([
        torch.tensor([0]),
        (target[2:] + target[1:-1] + target[:-2]) / 3,
        torch.tensor([1])
    ]).cummax(dim=-1).values

    return p_count, target, target_smoothed


if __name__ == '__main__':
    args = parse_args()
    print(args)

    # checkpoints = torch.load(args.src)
    filenames = glob(args.src)
    print(f'load checkpoints from {filenames}')

    checkpoints = [torch.load(filename) for filename in filenames]
    n_bins = args.n_bins
    bins = torch.linspace(0, 1, n_bins + 1)
    x = (bins[1:] + bins[:-1]) / 2

    X, Y_raw, Y = zip(*[
        get_ckpt_hist(ckpt, epoch, class_idx, score_name=args.score_name, bins=bins)
        for ckpt in checkpoints
        for epoch in range(1, len(ckpt))
        for class_idx in range(ckpt[1]['labels'].shape[1])
    ])

    X = torch.stack(X, dim=0)
    Y_raw = torch.stack(Y_raw, dim=0)
    Y = torch.stack(Y, dim=0)
    print(f'features: {X.shape}, targets: {Y.shape}')
    print(f'begin training ...')

    model = MyWeighter(n_bins)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-02, weight_decay=5e-5)

    model.train()
    for epoch in range(2000):
        Y_pred = model.fit(X)
        loss = mse_loss(Y_pred, Y, reduce='mean')
        if epoch % 100 == 0:
            print(f'epoch: {epoch:>3d}, loss: {loss.item():.2e}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), args.target)
    print(f'state_dict saved at {args.target}')
    print('done')
