import logging

from torch import nn
from torch.nn import functional as F
from typing import List, Union, Optional

from c_unet.utils.helpers.helpers import calc_upsampling_size


class Interpolate(nn.Module):
    """
    Down/up samples the input to the computed size 
    (mimicking the size computation of ConvTranspose3d).

    Args:
        - dilation (int): dilation that would haveen used by a transposed convolution
        - tconv_kernel_size (int): kernel size that would haveen used by a transposed convolution
        - tconv_stride (str, int): stride that would haveen used by a transposed convolution
        - tconv_padding (str, int): padding size that would haveen used by a transposed convolution
        - output_padding (str, int): output padding size that would haveen used by a transposed convolution
        - mode (str): interpolation mode, defaults to 'trilinear'
        - align_corners (Optional bool): specifies how to align corners
    """
    def __init__(self,
                 dilation: int = 1,
                 tconv_kernel_size: int = 3,
                 tconv_stride: Union[int, List[int]] = 2,
                 tconv_padding: Union[str, int] = 1,
                 output_padding: Union[str, int] = 1,
                 mode: str = 'trilinear',
                 align_corners: Optional[bool] = False):

        super(Interpolate, self).__init__()
        self.logger = logging.getLogger(__name__)

        self.dilation = dilation
        self.tconv_kernel_size = tconv_kernel_size
        self.tconv_stride = tconv_stride
        self.tconv_padding = tconv_padding
        self.output_padding = output_padding
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        _, _, h, w, d = x.shape

        # Computing sizes
        sizes = []
        for dim in (h, w, d):
            hs = calc_upsampling_size(dim, self.dilation,
                                      self.tconv_kernel_size,
                                      self.tconv_stride, self.tconv_padding,
                                      self.output_padding)
            sizes.append(hs)

        # Apply interpolate
        x = F.interpolate(x,
                          size=sizes,
                          mode=self.mode,
                          align_corners=self.align_corners)
        return x