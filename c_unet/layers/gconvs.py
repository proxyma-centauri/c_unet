import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from c_unet.layers.convs import ConvBlock
from c_unet.utils.dropout.GaussianDropout import GaussianDropout
from c_unet.utils.normalization.ReshapedBatchNorm import ReshapedBatchNorm
from c_unet.utils.normalization.ReshapedSwitchNorm import ReshapedSwitchNorm


class Gconv3d(nn.Module):
    """Performs a discretized convolution on SO(3)
 
    Args:
        group (str): Shorthand name representing the group to use
        expected_group_dim (int): Expected group dimension, it is 
            equal to the group dimension, except for the first Gconv of a series.
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Size of the kernel. Defaults to 3.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        stride (Union[int, List[int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[str, int], optional): Zero-padding added to all three sides of the input. Defaults to 1.
        dropout (float, optional): Value of dropout to use. Defaults to 0.1.
 
    Raises:
        ValueError: Invalid padding value
        ValueError: Unrecognized group
    """
    def __init__(self,
                 group: str,
                 expected_group_dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = "same",
                 dropout: float = 0.1):
        super(Gconv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.p = padding

        if group == "V":
            from c_unet.groups.V_group import V_group
            self.group = V_group()
            self.group_dim = self.group.group_dim
        elif group == "S4":
            from c_unet.groups.S4_group import S4_group
            self.group = S4_group()
            self.group_dim = self.group.group_dim
        elif group == "T4":
            from c_unet.groups.T4_group import T4_group
            self.group = T4_group()
            self.group_dim = self.group.group_dim
        else:
            raise ValueError(f"Group '{group}' is not recognized.")

        # Constants

        # W is the base filter. We rotate it 4 times for a p4 convolution over
        # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
        # one dimension in the channels.
        self.W = nn.Parameter(
            torch.Tensor(in_channels * expected_group_dim * out_channels,
                         kernel_size, kernel_size, kernel_size))

        torch.nn.init.xavier_uniform_(self.W)

        self.dropout = GaussianDropout(p=dropout)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape

        # Reshape and rotate the io filters 4 times. Each input-output pair is
        # rotated and stacked into a much bigger kernel
        x = x.reshape(bs, c * g, h, w, d)

        WN = self.group.get_Grotations(self.W.clone().to(x.device))
        WN = torch.stack(WN, 0)

        if g == 1:
            WN = WN.view(self.out_channels * self.group_dim, self.in_channels,
                         self.kernel_size, self.kernel_size, self.kernel_size)
        else:
            WN = WN.view(self.group_dim, self.out_channels, self.group_dim,
                         self.in_channels, self.kernel_size, self.kernel_size,
                         self.kernel_size)

            WN_shifted = self.group.G_permutation(WN)
            WN = torch.stack(WN_shifted, -1)

            WN = WN.view(self.out_channels * self.group_dim,
                         self.in_channels * self.group_dim, self.kernel_size,
                         self.kernel_size, self.kernel_size)

        # Gaussian dropout on the weights
        WN = self.dropout(WN)

        x = F.conv3d(x,
                     WN,
                     stride=self.stride,
                     padding=self.p,
                     dilation=self.dilation)
        _, _, h, w, d = x.shape
        x = x.view(bs, self.out_channels, self.group_dim, h, w, d)

        return x


class GconvBlock(nn.Module):
    """Applies a discretized convolution on SO(3) with optional bias addition,
    normalization and nonlinearity steps block

    Args:
        group (str): Shorthand name representing the group to use
        expected_group_dim (int): Expected group dimension, it is 
            equal to the group dimension, except for the first Gconv of a series.
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel. Defaults to 3.
        stride (Union[int, List[int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[str, int], optional): Zero-padding added to all three sides of the input. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(self,
                 group: str,
                 expected_group_dim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 bias: Optional[bool] = True,
                 nonlinearity: Optional[str] = "relu",
                 normalization: Optional[str] = "bn"):
        super(GconvBlock, self).__init__()
        self.bias = bias

        self.Gconv = Gconv3d(group, expected_group_dim, in_channels,
                             out_channels, kernel_size, dilation, stride,
                             padding, dropout)

        group_dim = self.Gconv.group_dim

        if bias:
            self.b = nn.Parameter(torch.full((1, 1), 0.01))

        other_modules = []

        if nonlinearity:
            if nonlinearity == "relu":
                other_modules.append(nn.ReLU(inplace=True))
            elif nonlinearity == "leaky-relu":
                other_modules.append(nn.LeakyReLU(inplace=True))
            elif nonlinearity == "elu":
                other_modules.append(nn.ELU(inplace=True))
            elif nonlinearity == "sigmoid":
                other_modules.append(nn.Sigmoid())
            elif nonlinearity == "softmax":
                other_modules.append(nn.Softmax(dim=1))
            else:
                raise ValueError(f"Invalid nonlinearity value: {nonlinearity}")

        if normalization:
            if normalization == "bn":
                other_modules.append(ReshapedBatchNorm(out_channels,
                                                       group_dim))
            elif normalization == "sn":
                other_modules.append(
                    ReshapedSwitchNorm(out_channels, group_dim))
            else:
                raise ValueError(
                    f"Invalid normalization value: {normalization}")

        self.OtherModules = nn.Sequential(*other_modules)

    def forward(self, x):
        x = self.Gconv(x)

        x.permute([0, 2, 1, 3, 4, 5])
        if self.bias:
            x += self.b
        self.OtherModules(x)
        x.permute([0, 2, 1, 3, 4, 5])

        return x


class GconvResBlock(nn.Module):
    """Applies a residual block with two discretized convolution on SO(3) with optional bias addition,
    normalization and nonlinearity steps

    Args:
        group (str): Shorthand name representing the group to use
        group_dim (int): Group dimension, it is 
            equal to the group dimension
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        is_first_conv (bool) : Boolean indicating whether the first convolution 
            of the residual block should have an expected_group_dim of 1.
        kernel_size (int): Size of the kernels. Defaults to 3.
        stride (Union[int, List[int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[str, int], optional): Zero-padding added to all three sides of the input. Defaults to 1.
        first_kernel_size (int): Overrides the size of the first kernel. Defaults to 3.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(self,
                 group: str,
                 group_dim: int,
                 in_channels: int,
                 inter_channels: int,
                 out_channels: int,
                 is_first_conv: bool = False,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1,
                 dilation: int = 1,
                 dropout: float = 0.1,
                 first_kernel_size: Optional[int] = None,
                 first_padding: Optional[int] = None,
                 bias: Optional[bool] = True,
                 nonlinearity: Optional[str] = "relu",
                 normalization: Optional[str] = "bn"):
        super(GconvResBlock, self).__init__()
        self.logger = logging.getLogger(__name__)

        expected_group_dim = 1 if is_first_conv else group_dim
        other_kernel_size = first_kernel_size if first_kernel_size is not None else kernel_size
        other_padding = first_padding if first_padding is not None else padding

        self.match_channels = GconvBlock(group,
                                         expected_group_dim,
                                         in_channels,
                                         out_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         dilation=1,
                                         dropout=0,
                                         bias=bias,
                                         nonlinearity=nonlinearity,
                                         normalization="")

        self.G_block_1 = GconvBlock(group,
                                    expected_group_dim,
                                    in_channels,
                                    inter_channels,
                                    other_kernel_size,
                                    stride,
                                    other_padding,
                                    dilation,
                                    dropout,
                                    bias=bias,
                                    nonlinearity=nonlinearity,
                                    normalization=normalization)
        self.G_block_2 = GconvBlock(
            group,
            group_dim,
            inter_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            dropout,
            bias=bias,
            nonlinearity="",  # No nonlinearity
            normalization=normalization)

        if nonlinearity == "leaky-relu":
            self.relu = nn.LeakyReLU(inplace=True)
        elif nonlinearity == "elu":
            self.relu = nn.ELU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.match_channels(x)
        y = self.G_block_1(x)
        y = self.G_block_2(y)
        y = self.relu(y + z)

        return y


class FinalGroupConvolution(nn.Module):
    """
    Add a final convolution with 1x1x1 kernel.

    Args:
        - stable_convolution (Module) : Stabilizing 5D convolution to perform before final convolution
        - group_dim : dimension of the sub group
        - out_channels (int) Number of output channels
        - final_activation (str) : Final activation layer
    """
    def __init__(self, stable_convolution: nn.Module, inter_channels: int,
                 out_channels: int, final_activation: str):
        super(FinalGroupConvolution, self).__init__()

        self.stable_convolution = stable_convolution
        self.reshaping_conv = ConvBlock(inter_channels,
                                        out_channels,
                                        kernel_size=1,
                                        padding="same",
                                        nonlinearity=final_activation,
                                        normalization="")

    def forward(self, x):
        # Reshaping input
        bs, c, g, h, w, d = x.shape
        x = x.reshape(bs, c * g, h, w, d)

        x = self.stable_convolution(x)

        # Removing group dimension
        x = self.reshaping_conv(x)
        return x
