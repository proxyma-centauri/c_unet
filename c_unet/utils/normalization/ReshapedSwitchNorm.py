from torch import nn
from c_unet.utils.normalization.SwitchNorm3d import SwitchNorm3d

class ReshapedSwitchNorm(nn.Module):
    """ Performs BatchNormalization through BatchNorm3d,
        after having reshaped the data into a 5d Tensor"""

    def __init__(self, 
                in_channels: int,
                group_dim: int,
                eps: float=1e-5,
                momentum: float=0.997,
                using_moving_average: bool=True,
                using_bn: bool=True,
                last_gamma: bool=False):
        super(ReshapedSwitchNorm, self).__init__()
        self.SwitchNormalization = SwitchNorm3d(in_channels * group_dim,
                                                eps,
                                                momentum,
                                                using_moving_average,
                                                using_bn,
                                                last_gamma)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = (self.SwitchNormalization(x)).view([bs, c, g, h, w, d])
        return x
