import torch
import torch.nn as nn
from typing import List, Optional, Union

from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock


class DecoderBlock(nn.Module):
    def __init__(self, 
                group: str,
                group_dim: int,
                # Channels
                out_channels: int, 
                # Kernel arguments
                kernel_size: int = 3,
                padding: Union[str, int] = 1,
                stride: Union[int, List[int]] = 1,
                # Transpose convolution
                tconv_size: int=3,
                tconv_stride: Union[int, List[int]] = 2,
                tconv_padding: Union[str, int] = 1,
                output_padding: Union[str, int] = 1,
                # Convolution arguments
                bias: bool = True,
                dilation: int = 1,
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn",
                # Model arguments
                model_depth: int=4,
                num_feat_maps: int = 16,
                num_conv_blocks: int = 2):
        super(DecoderBlock, self).__init__()

        self.num_feat_maps = num_feat_maps
        self.num_conv_blocks = num_conv_blocks

        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth - 2, -1, -1):
            feat_map_channels = 2 ** (depth + 1) * self.num_feat_maps

            # TODO : see how and if to adapt ConvTranspose3d to G-conv
            # TODO : see how to ajust kernel size, stride, etc to more situations while always allowing concatenations
            # TODO : check output_padding vs stride
            in_channels, out_channels = feat_map_channels * 4, feat_map_channels * 4
            self.deconv = nn.ConvTranspose3d(
                                            in_channels, 
                                            out_channels, 
                                            tconv_size,
                                            tconv_stride,
                                            tconv_padding,
                                            output_padding=output_padding)
            self.module_dict[f"deconv_{depth}"] = self.deconv

            for conv_nb in range(self.num_conv_blocks):
                if conv_nb == 0:
                    in_channels, out_channels = feat_map_channels * 6, feat_map_channels * 2
                    self.conv = ConvBlock(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        bias,
                                        dilation,
                                        nonlinearity,
                                        normalization)
                    self.module_dict[f"conv_{depth}_{conv_nb}"] = self.conv
                else:
                    in_channels, out_channels = feat_map_channels * 2, feat_map_channels * 2
                    self.conv = ConvBlock(in_channels,
                                        out_channels,
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
                self.final_conv = ConvBlock(in_channels=in_channels, out_channels=out_channels)
                self.module_dict["final_conv"] = self.final_conv

    def forward(self, x, down_sampling_features):
        """
        :param x: inputs
        :param down_sampling_features: feature maps from encoder path
        :return: output
        """
        for key, layer in self.module_dict.items():
            if key.startswith("deconv"):
                x = layer(x)
                x = torch.cat((down_sampling_features[int(key[-1])], x), dim=1)
            else:
                x = layer(x)
        return x