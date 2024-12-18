import os
import argparse
import torch
import sys
from tqdm import tqdm
from copy import deepcopy
from randaugment import RandAugment
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import normalize, cosine_similarity
from torchmetrics import MetricCollection
from torch.optim import lr_scheduler

from lib.dataset import build_dataset, supported_multilabel_datasets
from lib.transforms import RandomCutout
from lib.metric import (
    MultilabelHammingDistance,
    MultilabelRankingLoss,
    MultilabelAveragePrecision,
    MultilabelAveragePrecision2,
    MultilabelF1Score,
)
from lib.backbone import build_cnn_backbone, cnn_backbone_info, to_featuremap_backbone, GroupLinear
from lib.checkpointer import Checkpointer
from lib.utils import init_cuda_environment, str2bool, best_gpu
from lib.meter import AverageMeter
from lib.amp import GradScaler, autocast
from lib.loss import BceLoss, AsymmetricBceLoss as AslLoss, ece_loss

torch.multiprocessing.set_sharing_strategy('file_system')


@torch.no_grad()
def collect_outputs(model, train_loader):
    dataset = train_loader.dataset
    Y_score_clf = torch.zeros_like(dataset.labels).float()
    Y_score_proto = torch.zeros_like(dataset.labels).float()

    model.eval()
    for (img_w, img_s, y_true, y_partial, idxs) in train_loader:
        img_w = img_w.cuda()
        with autocast():
            y_score_clf, y_score_proto = model(img_w)

        Y_score_clf[idxs] = y_score_clf.cpu()
        Y_score_proto[idxs] = y_score_proto.cpu()

    return Y_score_clf, Y_score_proto


@torch.no_grad()
def evaluate_mAP(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes
    metric_clf = MultilabelAveragePrecision(num_labels=num_classes)
    metric_proto = MultilabelAveragePrecision(num_labels=num_classes)

    model.eval()
    for (img, y_true, y_noisy, idxs) in valid_loader:
        with autocast():
            y_score_clf, y_score_proto = model(img.cuda())

        metric_clf.update(y_score_clf.cpu(), y_true)
        metric_proto.update(y_score_proto.cpu(), y_true)

    mAP_clf = metric_clf.compute().item()
    mAP_proto = metric_proto.compute().item()

    return mAP_clf, mAP_proto


@torch.no_grad()
def evaluate_all_metrics(model, valid_loader):
    num_classes = valid_loader.dataset.num_classes

    metric = MetricCollection({
        'f1_score': MultilabelF1Score(num_classes),
        'h_loss': MultilabelHammingDistance(num_classes, threshold=0.5),
        'mAP': MultilabelAveragePrecision(num_classes),
        'mAP2': MultilabelAveragePrecision2(num_classes),
        'r_loss': MultilabelRankingLoss(num_classes),
    })

    model.eval()
    for (img, y_true, y_noisy, idxs) in valid_loader:
        with autocast():
            y_score_clf, y_score_proto = model(img.cuda())

        metric.update(y_score_clf.cpu(), y_true)

    return metric.compute()


def build_transform(name, *, image_size=224):
    if name == 'none':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    elif name == 'weak_augment':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    elif name == 'strong_augment':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            RandomCutout(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
        ])
    elif name == 'valid':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    else:
        raise ValueError(f'unsupported transform name ‘{name}’')


def build_weighter(name):

    class HardWeighter(nn.Module):

        def __init__(self, threshold):
            super().__init__()
            self.threshold = threshold

        def forward(self, y_score, y_partial):
            weight = torch.ones_like(y_score).float()
            mask = y_partial == 1
            weight[mask] = (y_score > self.threshold).float()[mask]
            return weight

    class SoftWeighter(nn.Module):

        def __init__(self, alpha):
            super().__init__()
            self.alpha = alpha

        def forward(self, y_score, y_partial):
            weight = torch.ones_like(y_score).float()
            mask = y_partial == 1
            weight[mask] = (y_score**self.alpha)[mask]
            return weight

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
                y = torch.cat([
                    torch.tensor([0], device=y.device), (y[2:] + y[1:-1] + y[:-2]) / 3,
                    torch.tensor([1], device=y.device)
                ]).cummax(dim=-1).values

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

    if name == 'hard':
        weighter = HardWeighter(threshold=0.7)
    elif name == 'soft':
        weighter = SoftWeighter(alpha=2.0)
    elif name == 'weighter' or name == 'weighter-optimal':
        weighter = MyWeighter(bins=20)
        weighter.load_state_dict(torch.load(args.weighter_ckpt), strict=True)
        print(f'weighter loaded from {args.weighter_ckpt}')
    elif name == 'identity':
        weighter = SoftWeighter(alpha=1.0)
    else:
        raise ValueError(f'unsupported weighter name ‘{name}’')
    return weighter


def build_model(backbone_name, num_classes, dim_embed, *, pretrained=True):

    class TransformerDecoderLayerWithoutSelfAttn(nn.Module):
        '''Similar to ML-Decoder
        '''

        def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True) -> None:
            super(TransformerDecoderLayerWithoutSelfAttn, self).__init__()
            self.norm1 = nn.LayerNorm(d_model, eps=1e-05)
            self.norm2 = nn.LayerNorm(d_model, eps=1e-05)
            self.norm3 = nn.LayerNorm(d_model, eps=1e-05)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
            self.activation = nn.ReLU()
            self.batch_first = batch_first

        def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                    memory_key_padding_mask=None):
            tgt = tgt + self.dropout1(tgt)
            tgt = self.norm1(tgt)
            tgt2, _ = self.multihead_attn(tgt, memory, memory)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)
            return tgt

    class MultilabelEncoderWithAttn(nn.Module):

        def __init__(self, num_classes, dim_feature, dim_embed):
            super().__init__()
            layer_decode = TransformerDecoderLayerWithoutSelfAttn(d_model=dim_embed, dim_feedforward=dim_embed * 4,
                                                                  dropout=0.1, nhead=8, batch_first=True)
            self.querys = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim_embed)
            self.requires_grad_(False)
            self.feature_projector = nn.Linear(dim_feature, dim_embed)
            self.query_projector = nn.Linear(dim_embed, dim_embed)
            self.decoder = nn.TransformerDecoder(layer_decode, num_layers=1)

        def forward(self, x):
            #  x: [batch_size, dim_feature, h, w]
            batch_size, _, h, w = x.shape

            x = x.flatten(2).transpose(1, 2)  # [batch_size, h*w, dim_feature]
            x = self.feature_projector(x)  # [batch_size, h*w, dim_embed]

            y = self.querys.weight  # [num_classes, dim_embed]
            y = self.query_projector(y)  # [num_classes, dim_embed]
            y = y.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, dim_embed]

            representations = self.decoder(y, x)  # [batch_size, num_classes, dim_embed]
            return representations

    class MyModel(nn.Module):

        def __init__(self, backbone_name, num_classes, dim_embed=512, pretrained=False):
            super().__init__()
            # featuremap backbone (output: [batch_size, dim_feature, h, w])
            cnn_backbone = build_cnn_backbone(name=backbone_name, pretrained=pretrained)
            dim_feature = cnn_backbone_info(cnn_backbone)['dim_featuremap']
            featuremap_backbone = to_featuremap_backbone(cnn_backbone)

            encoder = MultilabelEncoderWithAttn(num_classes, dim_feature=dim_feature, dim_embed=dim_embed)

            # for classification
            logit_projector = nn.Sequential(GroupLinear(num_classes, dim_embed, 1), nn.Flatten(1, 2))

            # for mutli-label prototype
            prototype_embeds = normalize(torch.randn((num_classes, dim_embed)).float(), dim=-1)
            prototype_embeds = nn.Parameter(prototype_embeds)
            prototype_embeds.requires_grad_(True)

            self.backbone_name = backbone_name
            self.num_classes = num_classes
            self.dim_embed = dim_embed
            self.pretrained = pretrained
            self.dim_feature = dim_feature

            self.backbone = featuremap_backbone
            self.encoder = encoder
            self.logit_projector = logit_projector
            self.prototype_embeds = prototype_embeds

        def forward(self, x):
            batch_size, _, H, W = x.shape  # [batch_size, 3, H, W]

            x = self.backbone(x)  # [batch_size, dim_feature, H2, W2]
            x = self.encoder(x)  # [batch_size, num_classer, dim_embed]

            # classification score
            clf_score = torch.sigmoid(self.logit_projector(x))  # [batch_size, num_classes]

            # prototype score
            proto = self.prototype_embeds
            proto = proto.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_classes, dim_embed]
            proto_score = 0.5 + 0.5 * cosine_similarity(x.float(), proto, dim=-1)  # [batch_size, num_classes]

            return clf_score.float(), proto_score.float()

        def __repr__(self):
            return f'MyModel(backbone_name={self.backbone_name}, ' \
                   f'num_classes={self.num_classes}, ' \
                   f'dim_embed={self.dim_embed}, ' \
                   f'pretrained={self.pretrained})'

    model = MyModel(backbone_name, num_classes, dim_embed=dim_embed, pretrained=pretrained)
    return model


def train_model(model, train_loader, valid_loader, *, args):
    model.cuda()

    weighter = build_weighter(args.weighting)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=5e-5)

    if args.criterion == 'bce':
        criterion = BceLoss(reduce='mean')
    elif args.criterion == 'asl':
        criterion = AslLoss(gamma_neg=4, gamma_pos=0, clip=0.05, no_focal_grad=False, reduce='mean')

    scaler = GradScaler()
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0)  # fixed learning rate
    dataset = train_loader.dataset

    Y_true = dataset.labels
    Y_partial = dataset.partial_labels
    Weight = torch.ones_like(Y_partial).float()

    def save_fn(score, model, epoch):
        checkpoint = deepcopy({
            'epoch': epoch,
            'mAP': score,
            'state_dict': {
                k: v.cpu()
                for k, v in model.state_dict().items()
            },
            'Y_score_clf': Y_score_clf,
            'Y_score_proto': Y_score_proto,
        })

        if epoch == 0:
            checkpoint['labels'] = deepcopy(dataset.labels)
            checkpoint['partial_labels'] = deepcopy(dataset.partial_labels)

        return checkpoint

    def load_fn(checkpoint):
        return checkpoint

    checkpointer = Checkpointer(patience=3, save_fn=save_fn, load_fn=load_fn, save_every_epoch=bool(args.export_ckpts))
    train_meter = AverageMeter()

    def train_one_epoch(coef_clf=1.0, coef_proto=0.0):
        train_meter.reset()
        model.train()
        for img_w, img_s, y_true, y_noisy, idxs in tqdm(train_loader, disable=not args.show_progress, leave=False):
            y_target = y_noisy.cuda()
            weight = Weight[idxs].cuda()

            with autocast():
                y_score_clf, y_score_proto = model(img_s.cuda())

            loss_clf = criterion(y_score_clf, y_target, weight=weight)
            loss_proto = criterion(y_score_proto, y_target, weight=weight)

            loss = torch.tensor(0.0).cuda()
            if coef_clf != 0:
                loss += coef_clf * loss_clf
            if coef_proto != 0:
                loss += coef_proto * loss_proto

            train_meter.update(loss_clf=loss_clf.item(), loss_proto=loss_proto.item(), loss=loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not scaler.is_scale_changed:
                scheduler.step()

    def display_one_epoch(mAP_clf, mAP_proto, stage_name, epoch):
        print(f'{stage_name} Epoch {epoch}: '
              f'lr {scheduler.get_last_lr()[0]:.1e}, '
              f'mAP@clf {mAP_clf:.2%}, mAP@proto {mAP_proto:.2%}, '
              f'{str(train_meter)}')
        return mAP_clf, mAP_proto

    # 1. warmup stage
    # ---------------
    for epoch in range(args.warmup_epochs):
        if checkpointer.early_stop():
            break

        train_one_epoch(coef_clf=args.coef_clf_warmup, coef_proto=args.coef_proto_warmup)
        Y_score_clf, Y_score_proto = collect_outputs(model, train_loader)
        mAP_clf, mAP_proto = evaluate_mAP(model, valid_loader)
        display_one_epoch(mAP_clf, mAP_proto, stage_name='Warmup', epoch=epoch)
        checkpointer.update(score=mAP_clf, model=model, epoch=epoch)

    ckpt = checkpointer.load()  # recover model parameters from best epoch
    model.load_state_dict(ckpt['state_dict'], strict=True)
    Y_score_clf, Y_score_proto = ckpt['Y_score_clf'], ckpt['Y_score_proto']

    print(f'load model from Warmup Epoch {ckpt["epoch"]:d} '
          f'with highest 1st-stage-mAP {ckpt["mAP"]:.2%}')
    checkpointer.reset()
    checkpointer.update(score=ckpt["mAP"], model=model, epoch=-1)

    # 2. disambiguation stage
    # -------------------------
    for epoch in range(args.epochs):
        if checkpointer.early_stop():
            break

        # X, Y_score_clf, Y_score_proto = collect_outputs(model, train_loader)
        Y_score_clf, Y_score_proto = collect_outputs(model, train_loader)
        if args.weighting != 'weighter-optimal':
            Weight = weighter(Y_score_clf, Y_partial).detach().float()
        else:
            Weight = weighter.forward_optimal(Y_score_clf, Y_partial, Y_true).detach().float()
        print(f'Confidence ECE: {ece_loss(Y_score_clf[Y_partial == 1], Y_true[Y_partial == 1], reduce="sum"):.4f}, '
              f'Weighting ECE: {ece_loss(Weight[Y_partial == 1], Y_true[Y_partial == 1], reduce="sum"):.4f}')
        Weight = torch.clamp(Weight * 1.2 - 0.1, 0, 1).float()

        train_one_epoch(coef_clf=args.coef_clf, coef_proto=args.coef_proto)
        mAP_clf, mAP_proto = evaluate_mAP(model, valid_loader)
        display_one_epoch(mAP_clf, mAP_proto, stage_name='Continual', epoch=epoch)
        checkpointer.update(score=mAP_clf, model=model, epoch=epoch)

    ckpt = checkpointer.load()
    model.load_state_dict(ckpt['state_dict'], strict=True)
    print(f'load model from Continual Epoch {ckpt["epoch"]:d} '
          f'with highest 2nd-stage-mAP {ckpt["mAP"]:.2%}')
    print(evaluate_all_metrics(model, valid_loader))

    if args.export_ckpts:
        # to cut memory cost of checkpoint file
        checkpoints = checkpointer.checkpoints
        for ckpt in checkpoints:
            del ckpt['state_dict']
        torch.save(checkpointer.checkpoints, args.export_ckpts)
        print(f'export checkpoints at {args.export_ckpts}')


def main(args):
    # initailize dataset, dataloader
    # ------------------------------

    print('loading dataset...', end=' ')
    train_dataset = build_dataset(name=args.dataset, divide='train', downsample_ratio=args.downsample_ratio,
                                  add_index=True, add_partial_noise=True, noise_rate=args.noise_rate, transforms=[
                                      build_transform('weak_augment'),
                                      build_transform('strong_augment')
                                  ], flatten=True)
    valid_dataset = build_dataset(name=args.dataset, divide='test', downsample_ratio=args.downsample_ratio,
                                  add_index=True, add_partial_noise=True, noise_rate=args.noise_rate, transforms=[
                                      build_transform('valid')
                                  ], flatten=True)
    print(train_dataset)

    args.num_samples = train_dataset.num_samples
    args.num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=(len(train_dataset) < 10000),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=(len(valid_dataset) < 10000),
    )

    print('loading model...', end=' ')
    model = build_model(backbone_name=args.backbone, num_classes=train_dataset.num_classes, dim_embed=args.dim_embed,
                        pretrained=args.pretrained)
    print(model)

    print('start training')
    train_model(model, train_loader, valid_loader, args=args)

    print('done.')


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='PML-CD`',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add = parser.add_argument

    group = parser.add_argument_group('model setting')
    group.add = group.add_argument
    group.add('--backbone', default='resnet101', type=str, choices=[
        'resnet50', 'resnet101', 'tresnet_l', 'tresnet_xl', 'tresnet_v2_l'
    ])
    group.add('--pretrained', default=True, type=str2bool)
    group.add('--dim-embed', default=512, type=int)

    group = parser.add_argument_group('dataset setting')
    group.add = group.add_argument
    group.add('--dataset', default='rand', type=str, choices=supported_multilabel_datasets())
    group.add('--noise-rate', default=0.0, type=float)
    group.add('--image-size', default=224, type=int, choices=[224, 448])
    group.add('--downsample-ratio', default=1.0, type=float)

    # disambigauation hyper-parameters
    group = parser.add_argument_group('coef hyper-parameters')
    group.add = group.add_argument
    group.add('--coef-clf', type=float, default=1.0)
    group.add('--coef-proto', type=float, default=1.0)  # validated
    group.add('--coef-clf-warmup', type=float, default=1.0)
    group.add('--coef-proto-warmup', type=float, default=0.0)

    # weighting hyper-parameters
    group.add('--weighting', type=str, default='weighter', choices=[
        'hard', 'soft', 'weighter', 'identity', 'weighter-optimal'
    ])
    group.add('--weighter-ckpt', type=str, default='weighter.ckpt')

    group = parser.add_argument_group('training setting')
    group.add = group.add_argument
    group.add('--batch-size', default=32, type=int)
    group.add('--lr', help='learning rate', default=1e-04, type=float)
    group.add('--warmup-epochs', default=20, type=int)
    group.add('--epochs', default=20, type=int)
    group.add('--criterion', default='bce', type=str, choices=['bce', 'asl'])

    group = parser.add_argument_group('misc setting')
    group.add = group.add_argument
    group.add('--seed', help='random seed', default=123, type=int)
    group.add('--gpu', type=str, default=str(best_gpu()))
    group.add('--show-progress', type=str2bool, default=(os.isatty(sys.stdout.fileno())))
    group.add('--export-ckpts', default='', type=str)

    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    print(f'{args = }')

    init_cuda_environment(seed=args.seed, device=args.gpu)

    main(args)
