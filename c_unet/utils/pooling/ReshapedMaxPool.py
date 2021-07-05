import logging

from einops import rearrange, reduce
from torch import nn
from typing import List, Union


class ReshapedMaxPool(nn.Module):
    """ Performs MaxPooling through MaxPool3d,
        after having reshaped the data into a 5d Tensor"""
    def __init__(self,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1):

        super(ReshapedMaxPool, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.MaxPool = nn.MaxPool3d(kernel_size, stride, padding)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = self.MaxPool(x)

        # New shape
        _, _, new_h, new_w, new_d = x.shape
        x = x.view([bs, c, g, new_h, new_w, new_d])
        return x


class GMaxPool3d(nn.Module):
    """Pool over channels"""
    def __init__(self,
                 num_group: int = 4,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1):
        super(GMaxPool3d, self).__init__()
        self.g = num_group
        self.pool = nn.MaxPool3d(kernel_size, stride, padding)

    def forward(self, x):
        x = rearrange(x, "b c g h w d -> (b g) c h w d")
        x = self.pool(x)
        return rearrange(x, "(b g) c h w d -> b c g h w d", g=self.g)


class GPool3d(nn.Module):
    """Pool over channels using einops.
    
    pool_over: `c`, `g`, `cg` or `hwd`
    reduction: `max` or `mean`
    """
    def __init__(self,
                 pool_over: str = "c",
                 reduction: str = "mean",
                 reduction_factor: int = 2):
        super(GPool3d, self).__init__()
        self.pool_over = pool_over
        self.reduction = reduction
        self.reduction_factor = reduction_factor

    def forward(self, x):
        if self.pool_over == "c":
            return reduce(x,
                          "b (c c2) g h w d -> b c g h w d",
                          reduction=self.reduction,
                          c2=self.reduction_factor)
        elif self.pool_over == "g":
            return reduce(x,
                          "b c (g g2) h w d -> b c g h w d",
                          reduction=self.reduction,
                          g2=self.reduction_factor)
        elif self.pool_over == "cg":
            return reduce(x,
                          "b (c c2) (g g2) h w d -> b c g h w d",
                          reduction=self.reduction,
                          c2=self.reduction_factor,
                          g2=self.reduction_factor)
        elif self.pool_over == "hwd":
            return reduce(x,
                          "b c g (h h2) (w w2) (d d2) -> b c g h w d",
                          reduction=self.reduction,
                          h2=self.reduction_factor,
                          w2=self.reduction_factor,
                          d2=self.reduction_factor)
