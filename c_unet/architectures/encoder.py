from c_unet.architectures.dilated_dense import DilatedDenseBlock
import logging
from typing import List, Optional, Union
import torch.nn as nn

from c_unet.layers.gconvs import GconvResBlock
from c_unet.layers.convs import ConvResBlock
from c_unet.utils.pooling.GPool3d import GPool3d


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
        - pool_reduction : Type of pooling for G-CNNs. Defualts to "max".
        - pool_factor : For G-cnns, reduction factor of the pooling. Defaults to 2.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - root_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 32.
        - num_conv_blocks (int): Number of convolutions per block at specific depth. Defaults to 2.

        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(
            self,
            # Channels
            in_channels: int,
            # Kernel arguments
            kernel_size: int = 3,
            stride: Union[int, List[int]] = 1,
            padding: Union[str, int] = "same",
            # Pooling
            pool_size: Optional[int] = 2,
            pool_stride: Optional[Union[str, int]] = 2,
            pool_padding: Optional[Union[str, int]] = 0,
            pool_reduction: Optional[str] = "mean",
            pool_factor: Optional[int] = 2,
            # Convolution arguments
            dropout: Optional[bool] = 0.1,
            bias: bool = True,
            dilation: int = 1,
            nonlinearity: Optional[str] = "relu",
            normalization: Optional[str] = "bn",
            # Model
            model_depth: int = 4,
            root_feat_maps: int = 32,
            # Group arguments (by default, no group)
            group: Union[str, None] = None,
            group_dim: int = 0):
        super(EncoderBlock, self).__init__()

        self.root_feat_maps = root_feat_maps
        self.logger = logging.getLogger(__name__)

        self.module_dict = nn.ModuleDict()

        # U-net structure
        for depth in range(model_depth):
            feat_map_channels = 2**(depth + 1) * self.root_feat_maps

            if group:
                is_first_conv = True if (depth == 0) else False
                self.module_dict[f"conv_block_{depth}"] = GconvResBlock(
                    group,
                    group_dim,
                    in_channels,
                    feat_map_channels,
                    feat_map_channels,
                    is_first_conv,
                    kernel_size,
                    stride,
                    padding,
                    dilation=dilation,
                    dropout=dropout,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    normalization=normalization)
            else:
                self.module_dict[f"conv_block_{depth}"] = ConvResBlock(
                    in_channels,
                    feat_map_channels,
                    feat_map_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=bias,
                    dilation=dilation,
                    nonlinearity=nonlinearity,
                    normalization=normalization)

            in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2

            if depth == 1:
                self.dilated_dense = DilatedDenseBlock(
                    in_channels, feat_map_channels, in_channels, kernel_size,
                    stride, dropout, bias, nonlinearity, normalization, 3, 2,
                    group, group_dim)

            if depth == model_depth - 1:
                break
            else:
                if group:
                    self.module_dict[f"max_pooling_{depth}"] = GPool3d(
                        pool_over="hwd",
                        reduction=pool_reduction,
                        reduction_factor=pool_factor)
                else:
                    self.module_dict[f"max_pooling_{depth}"] = nn.MaxPool3d(
                        kernel_size=pool_size,
                        stride=pool_stride,
                        padding=pool_padding)

    def forward(self, x):
        """
        Args:
            - x: input feature map
        Returns:
            - (output feature map, down_sampling_features): tuple with the last feature map
                of the encoder path and the feature maps from the encoder path
        """
        down_sampling_features = []
        for key, layer in self.module_dict.items():
            if key.startswith("conv"):
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")
                if key.endswith("1"):  # Layer 1
                    x = self.dilated_dense(x)
                down_sampling_features.append(x)
            elif key.startswith("max_pooling"):
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")

        return x, down_sampling_features