import logging
import torch
import torch.nn as nn
from typing import List, Optional, Union

from src.utils.interpolation.ReshapedInterpolate import ReshapedInterpolate
from src.utils.interpolation.Interpolate import Interpolate

from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock


class DecoderBlock(nn.Module):
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
                # self.upsample = nn.ConvTranspose3d(in_channels,
                #                                 inter_channels,
                #                                 tconv_kernel_size,
                #                                 tconv_stride,
                #                                 tconv_padding,
                #                                 output_padding=output_padding)
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
                    self.final_conv = GconvBlock(group,
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
                else:
                    self.final_conv = ConvBlock(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        bias,
                                        dilation,
                                        nonlinearity,
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