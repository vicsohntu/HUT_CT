import torch.nn.functional as F
import torch
import numpy as np
# for metric code: https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py

import torch.nn as nn

from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, reduction='none'):
        super().__init__(weight=alpha, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss

cross_entropy = F.cross_entropy

def hard_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)
    bg = (target == 0)
    neg = mtx[bg]
    pos = mtx[~bg]
    Np, Nn = pos.numel(), neg.numel()
    pos = pos.sum()
    k = min(Np*alpha, Nn)
    if k > 0:
        neg, _ = torch.topk(neg, int(k))
        neg = neg.sum()
    else:
        neg = 0.0
    loss = (pos + neg)/ (Np + k)
    return loss


def hard_per_im_cross_entropy(output, target, alpha=3.0):
    n, c = output.shape[:2]
    output = output.view(n, c, -1)
    target = target.view(n, -1)
    mtx = F.cross_entropy(output, target, reduce=False)
    pos = target > 0
    num_pos = pos.long().sum(dim=1, keepdim=True)
    loss = mtx.clone().detach()
    loss[pos] = 0
    _, loss_idx = loss.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_neg = torch.clamp(alpha*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg
    return mtx[neg + pos].mean()

def mean_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    pos = pos.mean() if pos.numel() > 0 else 0
    neg = neg.mean() if pos.neg() > 0 else 0

    loss = (neg * alpha + pos)/(alpha + 1.0)
    return loss




eps = 1e-8
def dice(output, target):
    eps = 1e-8
    num = 2*(output*target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def dice_score(output, target):
    eps = 1e-8
    num = 2*(output*target).sum() + eps
    den = output.sum() + target.sum() + eps
    return num/den


def cross_entropy_dice(output, target, batch, weight=1.0):
    class_wgt=np.asarray([0.25659472, 45.465614, 16.543337, 46.11155])
    class_wgt=class_wgt/class_wgt.sum()
    loss=0.0
    output = F.softmax(output, dim=1)
    for i in range(batch):
        for c in range(0, 4):
            o = output[i, c]
            a = (target[i]==c)
            t = a.float()
            loss += class_wgt[c]*dice(o, t)
    return loss

def dice_loss_wgt(output, target, batch):
    loss = 0.0
    class_wgt=np.asarray([0.1, 0.9])
    output = F.softmax(output, dim=1)
    for i in range(batch):
        for c in range(0, 2):
            o = output[i, c]
            a = (target==c)
            t = a.float()
            loss += class_wgt[c]*dice(o, t)
    return loss

def weighted_dice_loss(output, target, class_weights = torch.tensor([0.1,0.9]), softmax=True):
    if softmax:
        output = F.softmax(output, dim=1)
    eps = 1e-5
    num = 2*(class_weights*(output*target).sum(axis=(0,1))).sum() + eps
    den = (class_weights*(output.sum(axis=(0,1)) + target.sum(axis=(0,1)))).sum() + eps
    return 1.0 - num/den

# in original paper: class 3 is ignored
# https://github.com/MIC-DKFZ/BraTS2017/blob/master/dataset.py#L283
# dice score per image per positive class, then aveg
def dice_per_im(output, target):
    eps = 0.1
    n = output.shape[0]
    output = output.view(n, -1)
    target = target.view(n, -1)
    num = 2*(output*target).sum(1) + eps
    den = output.sum(1) + target.sum(1) + eps
    return 1.0 - (num/den).mean()

def cross_entropy_dice_per_im(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 5):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice_per_im(o, t)

    return loss
