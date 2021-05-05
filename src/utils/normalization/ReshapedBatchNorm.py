import torch
from torch import nn

class ReshapedBatchNorm(nn.Module):
    """ Performs BatchNormalization through BatchNorm3d,
        after having reshaped the data into a 5d Tensor"""

    def __init__(self):
        super(ReshapedBatchNorm, self).__init__()

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        BatchNormalization = nn.BatchNorm3d(c * g)
        x = (BatchNormalization(x)).view([bs, c, g, h, w, d])
        return x
