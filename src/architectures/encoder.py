import logging
from typing import List, Optional, Union
import torch.nn as nn

from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock
from src.utils.pooling.ReshapedMaxPool import ReshapedMaxPool


class EncoderBlock(nn.Module):
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
                # Kernel arguments
                kernel_size: int = 3,
                stride: Union[int, List[int]] = 1,
                padding: Union[str, int] = 1,
                # Pooling
                pool_size: int=2,
                pool_stride: Union[int, List[int]] = 2,
                pool_padding: Union[str, int] = 0,
                # Convolution arguments
                dropout: Optional[bool] = 0.1,
                bias: bool = True,
                dilation: int = 1,
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn",
                # Model
                model_depth: int=4,
                root_feat_maps: int = 16,
                num_conv_blocks: int = 2,
                # Group arguments (by default, no group)
                group: Union[str, None]=None,
                group_dim: int=0):
        super(EncoderBlock, self).__init__()

        self.root_feat_maps = root_feat_maps
        self.num_conv_blocks = num_conv_blocks
        self.logger = logging.getLogger(__name__)

        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps

            for conv_nb in range(self.num_conv_blocks):
                if group:
                    expected_group_dim = 1 if (depth == 0 and conv_nb == 0) else group_dim
                    self.conv_block = GconvBlock(group,
                                            expected_group_dim,
                                            in_channels,
                                            feat_map_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            dilation,
                                            dropout,
                                            bias,
                                            nonlinearity,
                                            normalization)
                else: # TODO check order of arguments
                    self.conv_block = ConvBlock(in_channels,
                                            feat_map_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            bias,
                                            dilation,
                                            nonlinearity,
                                            normalization)
                self.module_dict[f"conv_{depth}_{conv_nb}"] = self.conv_block

                in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2

            if depth == model_depth - 1:
                break
            else:
                if group:
                    self.pooling = ReshapedMaxPool(kernel_size=pool_size, stride=pool_stride, padding=pool_padding)
                else:
                    self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=pool_stride, padding=pool_padding)
                
                self.module_dict[f"max_pooling_{depth}"] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for key, layer in self.module_dict.items():
            if key.startswith("conv"):
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")
                if key.endswith("1"):
                    down_sampling_features.append(x)
            elif key.startswith("max_pooling"):
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")

        return x, down_sampling_features