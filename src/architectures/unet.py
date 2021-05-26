import logging
from typing import List, Optional, Union

import torch.nn as nn

from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock
from src.architectures.decoder import DecoderBlock
from src.architectures.encoder import EncoderBlock

class Unet(nn.Module):

    def __init__(self,
                # Group arguments
                group: Union[str, None],
                group_dim: int,
                # Channels arguments
                in_channels: int,
                out_channels: int,
                # Pooling
                pool_size: int=2,
                pool_stride: Union[int, List[int]] = 2,
                pool_padding: Union[str, int] = 0,
                # Transpose convolutions arguments
                tconv_kernel_size: int=3,
                tconv_stride: Union[int, List[int]] = 2,
                tconv_padding: Union[str, int] = 1,
                output_padding: Union[str, int] = 1,
                # Convolutional arguments
                dropout: Optional[bool] = 0.1,
                stride: Union[int, List[int]] = 1,
                padding: Union[str, int] = 1,
                kernel_size: int = 3,
                bias: bool = True,
                dilation: int = 1,
                # Additional layers
                nonlinearity: Optional[str] = "relu",
                normalization: Optional[str] = "bn",
                # Architecture arguments
                model_depth=4,
                root_feat_maps: int = 16,
                num_feat_maps: int = 16,
                num_conv_blocks: int = 2,
                final_activation="sigmoid"):
        super(Unet, self).__init__()

        self.logger = logging.getLogger(__name__)
        self.encoder = EncoderBlock(in_channels=in_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    pool_size=pool_size,
                                    pool_stride=pool_stride,
                                    pool_padding=pool_padding,
                                    dropout=dropout,
                                    bias=bias,
                                    dilation=dilation,
                                    nonlinearity=nonlinearity,
                                    normalization=normalization,
                                    model_depth=model_depth,
                                    root_feat_maps=root_feat_maps,
                                    num_conv_blocks=num_conv_blocks,
                                    group=group,
                                    group_dim=group_dim)

        self.decoder = DecoderBlock(out_channels=out_channels, 
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    tconv_kernel_size=tconv_kernel_size,
                                    tconv_stride=tconv_stride,
                                    tconv_padding=tconv_padding,
                                    output_padding=output_padding,
                                    dropout=dropout,
                                    bias=bias,
                                    dilation=dilation,
                                    nonlinearity=nonlinearity,
                                    normalization=normalization,
                                    model_depth=model_depth,
                                    num_feat_maps=num_feat_maps,
                                    num_conv_blocks=num_conv_blocks,
                                    group=group,
                                    group_dim=group_dim,)

        # TODO : make this group compatible
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        self.logger.debug(f"Final output shape: {x.shape}")
        return x
