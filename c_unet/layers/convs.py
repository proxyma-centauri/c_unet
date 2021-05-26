
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from typing import List, Optional, Union

from c_unet.utils.helpers.helpers import conv3d
from c_unet.utils.normalization.SwitchNorm3d import SwitchNorm3d

class ConvBlock(nn.Module):
    """Applies a 3D convolution with optional normalization and nonlinearity steps block

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel. Defaults to 3.
        stride (Union[int, List[int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[str, int], optional): Zero-padding added to all three sides of the input. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 nonlinearity: Optional[str] = "relu",
                 normalization: Optional[str] = "bn"):
        super(ConvBlock, self).__init__()

        modules = [
            conv3d(in_channels, out_channels, kernel_size, stride, padding,
                   bias, dilation)
        ]

        # ! WARNING: You'll end up using a batch size of 1, we need another
        # ! normalization layer (e.g. switchnorm).
        if normalization:
            if normalization == "bn":
                modules.append(nn.BatchNorm3d(out_channels))
            elif normalization == "sn":
                modules.append(SwitchNorm3d(out_channels))
            else:
                raise ValueError(
                    f"Invalid normalization value: {normalization}")

        if nonlinearity:
            if nonlinearity == "relu":
                modules.append(nn.ReLU(inplace=True))
            elif nonlinearity == "sigmoid":
                modules.append(nn.Sigmoid())
            elif nonlinearity == "softmax":
                modules.append(nn.Softmax(dim=1))
            else:
                raise ValueError(f"Invalid nonlinearity value: {nonlinearity}")

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)
