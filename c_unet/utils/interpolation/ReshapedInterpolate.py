import logging

from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional

from c_unet.utils.helpers.helpers import calc_upsampling_size
from c_unet.utils.interpolation.Interpolate import Interpolate


class ReshapedInterpolate(nn.Module):
    """
    Down/up samples the input to the computed size 
    (mimicking the size computation of ConvTranspose3d), while reshaping.
    """

    def __init__(self,
                dilation: int = 1,
                tconv_kernel_size: int=3,
                tconv_stride: Union[int, List[int]] = 2,
                tconv_padding: Union[str, int] = 1,
                output_padding: Union[str, int] = 1, 
                mode: str='trilinear', 
                align_corners: Optional[bool]=None):

        super(ReshapedInterpolate, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.interpolate = Interpolate(
                                dilation,
                                tconv_kernel_size,
                                tconv_stride,
                                tconv_padding,
                                output_padding,
                                mode,
                                align_corners)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = self.interpolate(x)   
        _, _, new_h, new_w, new_d = x.shape
        x = x.view([bs, c, g, new_h, new_w, new_d])
        return x