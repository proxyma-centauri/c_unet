from torch import nn
from typing import List, Union
from einops import rearrange


class ReshapedAvgPool(nn.Module):
    """ Performs AveragePooling through AvgPool3d,
        after having reshaped the data into a 5d Tensor"""
    def __init__(self,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1):

        super(ReshapedAvgPool, self).__init__()
        self.AvgPool = nn.AvgPool3d(kernel_size, stride, padding)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = self.AvgPool(x)

        # New shape
        _, _, new_h, new_w, new_d = x.shape
        x = x.view([bs, c, g, new_h, new_w, new_d])
        return x


class GAvgPool3d(nn.Module):
    """AvgPool over c*g"""
    def __init__(self,
                 num_group: int = 4,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1):
        super(GAvgPool3d, self).__init__()
        self.g = num_group
        self.pool = nn.AvgPool3d(kernel_size, stride, padding)

    def forward(self, x):
        x = rearrange(x, "b c g h w d -> b (c g) h w d")
        x = self.pool(x)
        return rearrange(x, "b (c g) h w d -> b c g h w d", g=self.g)