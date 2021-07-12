"""
CrossentropyND and TopKLoss are from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/training/loss_functions/ND_Crossentropy.py
"""

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np


class CrossentropyND(torch.nn.CrossEntropyLoss):
    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(
                inp.shape
        ):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1, )

        return super(CrossentropyND, self).forward(inp, target)


class TopKLoss(CrossentropyND):
    def __init__(self, weight=None, ignore_index=-100, k=10):
        self.k = k
        super(TopKLoss, self).__init__(weight,
                                       False,
                                       ignore_index,
                                       reduce=False)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape)
        res, _ = torch.topk(res.view((-1, )),
                            int(num_voxels * self.k / 100),
                            sorted=False)
        return res.mean()


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(
                inp.shape
        ):  # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1, )
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)