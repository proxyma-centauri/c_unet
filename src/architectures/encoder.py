import torch.nn as nn
from typing import List, Optional, Union

from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock


class EncoderBlock(nn.Module):
    def __init__(self, 
                in_channels: int, 
                kernel_size: int = 3,
                model_depth: int=4,
                pool_size: int=2,
                stride: Union[int, List[int]] = 1,
                padding: Union[str, int] = 1,
                bias: bool = True,
                dilation: int = 1,
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn"):
        super(EncoderBlock, self).__init__()

        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        self.module_dict = nn.ModuleDict()

        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps

            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                # print(in_channels, feat_map_channels)
                self.conv_block = ConvBlock(in_channels,
                                            feat_map_channels,
                                            kernel_size,
                                            stride,
                                            padding,
                                            bias,
                                            dilation,
                                            nonlinearity,
                                            normalization)
                self.module_dict[f"conv_{depth}_{i}"] = self.conv_block

                in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2

            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for key, layer in self.module_dict.items():
            if key.startswith("conv"):
                x = layer(x)
                print(key, x.shape)
                if key.endswith("1"):
                    down_sampling_features.append(x)
            elif key.startswith("max_pooling"):
                x = layer(x)
                print(key, x.shape)

        return x, down_sampling_features