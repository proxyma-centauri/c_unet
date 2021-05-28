import logging
from typing import List, Optional, Union
import torch.nn as nn

from c_unet.layers.gconvs import GconvResBlock
from c_unet.layers.convs import ConvResBlock
from c_unet.utils.pooling.ReshapedMaxPool import ReshapedMaxPool


class DilatedDenseBlock(nn.Module):
    """Encoding path of a U-Net architecture

    Args:
        - in_channels (int): Number of input channels   

        - kernel_size (int): Size of the kernel. Defaults to 3.
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.
        - padding (Union[str, int]): Zero-padding added to all three sides of the input. Defaults to 1.

        - pool_size (int): Size of the pooling kernel. Defaults to 2.
        - pool_stride (Union[int, List[int]]): Stride of the pooling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at pooling. Defaults to 0.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - root_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 16.
        - num_conv_blocks (int): Number of convolutions per block at specific depth. Defaults to 2.

        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(self, 
                # Channels
                in_channels: int,
                inter_channels: int,
                out_channels: int,
                # Kernel arguments
                kernel_size: int = 3,
                stride: Union[int, List[int]] = 1,
                # Convolution arguments
                dropout: Optional[bool] = 0.1,
                bias: bool = True,
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn",
                # Model
                num_steps_block: int = 3,
                dilation_increase_step: int = 2,
                # Group arguments (by default, no group)
                group: Union[str, None]=None,
                group_dim: int=0):
        super(DilatedDenseBlock, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.module_list = []

        for step_nb in range(num_steps_block):

            for _ in range(dilation_increase_step):
                dilation = 2**(step_nb)
                padding = (dilation * (kernel_size - 1) + 1) // 2

                if group:
                    self.conv_block = GconvResBlock(group,
                                            group_dim,
                                            in_channels,
                                            inter_channels,
                                            out_channels,
                                            False,
                                            kernel_size=kernel_size,
                                            first_kernel_size=1,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            dropout=dropout,
                                            first_padding=0,
                                            bias=bias,
                                            nonlinearity=nonlinearity,
                                            normalization=normalization)
                else:
                    self.conv_block = ConvResBlock(in_channels,
                                            inter_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            first_kernel_size=1,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            first_padding=0,
                                            dilation=dilation,
                                            nonlinearity=nonlinearity,
                                            normalization=normalization)
                self.module_list.append(self.conv_block)
        
        self.dilated_dense = nn.Sequential(*self.module_list)


    def forward(self, x):
        return self.dilated_dense(x)