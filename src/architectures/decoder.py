import logging
import torch
import torch.nn as nn
from typing import List, Optional, Union

from src.utils.interpolation.ReshapedInterpolate import ReshapedInterpolate
from src.utils.interpolation.Interpolate import Interpolate

from src.layers.gconvs import GconvResBlock, GconvBlock, FinalGroupConvolution
from src.layers.convs import ConvBlock


class DecoderBlock(nn.Module):
    """Encoding path of a U-Net architecture

    Args:
        - out_channels (int): Number of output channels   

        - kernel_size (int): Size of the kernel. Defaults to 3.
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.
        - padding (Union[str, int]): Zero-padding added to all three sides of the input. Defaults to 1.

        - tconv_kernel_size (int): Size of the kernel. Defaults to 3.
        - tconv_stride (Union[int, List[int]]): Stride of the upsampling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at upsampling. Defaults to 1.
        - output_padding (Union[str, int]): Additional size added to one side of each dimension in the output shape. Defaults to 1.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - num_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 16.
        - num_conv_blocks (int): Number of convolutions per block at specific depth. Defaults to 2.
        - final_activation (str): Name of the final activation to use. Defaults to sigmoid.

        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """
    def __init__(self,
                # Channels
                out_channels: int, 
                # Kernel arguments
                kernel_size: int = 3,
                padding: Union[str, int] = 1,
                stride: Union[int, List[int]] = 1,
                # Transpose convolution
                tconv_kernel_size: int=3,
                tconv_stride: Union[int, List[int]] = 2,
                tconv_padding: Union[str, int] = 1,
                output_padding: Union[str, int] = 1,
                # Convolution arguments
                dropout: Optional[bool] = 0.1,
                bias: bool = True,
                dilation: int = 1,
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn",
                # Model arguments
                model_depth: int=4,
                num_feat_maps: int = 16,
                num_conv_blocks: int = 2,
                final_activation: str ="sigmoid",
                # Group arguments (by default, no group)
                group: Union[str, None]=None,
                group_dim: int=0):
        super(DecoderBlock, self).__init__()

        self.num_feat_maps = num_feat_maps
        self.num_conv_blocks = num_conv_blocks
        self.logger = logging.getLogger(__name__)

        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps

            # TODO : see how to ajust kernel size, stride, etc to more situations while always allowing concatenations
            # TODO : check output_padding vs stride
            in_channels, inter_channels = feat_map_channels * 4, feat_map_channels * 4

            if group:
                self.upsample = ReshapedInterpolate(dilation,
                                            tconv_kernel_size,
                                            tconv_stride,
                                            tconv_padding,
                                            output_padding)
            else:
                self.upsample = Interpolate(dilation,
                                        tconv_kernel_size,
                                        tconv_stride,
                                        tconv_padding,
                                        output_padding)
            
            self.module_dict[f"upsample_{depth}"] = self.upsample

            for conv_nb in range(self.num_conv_blocks):

                # Multiplier factor for channels
                multiplier = 2
                if conv_nb == 0:
                    multiplier = 6

                in_channels, inter_channels = feat_map_channels * multiplier, feat_map_channels * 2
                if group:
                    self.conv = GconvBlock(group,
                                        group_dim,
                                        in_channels,
                                        inter_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        dropout,
                                        bias,
                                        nonlinearity,
                                        normalization)
                else:
                    self.conv = ConvBlock(in_channels,
                                    inter_channels,
                                    kernel_size,
                                    stride,
                                    padding,
                                    bias,
                                    dilation,
                                    nonlinearity,
                                    normalization)
                self.module_dict[f"conv_{depth}_{conv_nb}"] = self.conv

            if depth == 0:
                in_channels = feat_map_channels * 2
                if group:
                    group_final_conv = GconvBlock(group,
                                        group_dim,
                                        in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        dropout,
                                        bias,
                                        nonlinearity,
                                        normalization)
                    self.final_conv = FinalGroupConvolution(group_final_conv,
                                                    group_dim,
                                                    out_channels,
                                                    final_activation)
                else:
                    self.final_conv = ConvBlock(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        bias,
                                        dilation,
                                        final_activation,
                                        normalization)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for key, layer in self.module_dict.items():
            if key.startswith("upsample"):
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")
                x = torch.cat((down_sampling_features[int(key[-1])], x), dim=1)
            else:
                x = layer(x)
                self.logger.debug(f"{key}, {x.shape}")
        return x