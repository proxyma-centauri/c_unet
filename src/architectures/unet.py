import logging
import torch.nn as nn

from typing import List, Optional, Union

from src.architectures.decoder import DecoderBlock
from src.architectures.encoder import EncoderBlock

class Unet(nn.Module):
    """ U-net architecture, that can be used either with normal convolutions, or with group convolutions.
    The available groups are defined in equiHippo/groups

    Args:
        - group (str): Shorthand name representing the group to use
        - group_dim (int): Group dimension

        - in_channels (int): Number of input channels   
        - out_channels (int): Number of output channels   

        - pool_size (int): Size of the pooling kernel. Defaults to 2.
        - pool_stride (Union[int, List[int]]): Stride of the pooling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at pooling. Defaults to 0.

        - tconv_kernel_size (int): Size of the kernel. Defaults to 3.
        - tconv_stride (Union[int, List[int]]): Stride of the upsampling. Defaults to 2.
        - pool_padding (Union[str, int]): Zero-padding added to all three sides of the input at upsampling. Defaults to 1.
        - output_padding (Union[str, int]): Additional size added to one side of each dimension in the output shape. Defaults to 1.

        - dropout (float, optional) : Value of dropout to use. Defaults to 0.1
        - stride (Union[int, List[int]]): Stride of the convolution. Defaults to 1.
        - padding (Union[str, int]): Zero-padding added to all three sides of the input. Defaults to 1.
        - kernel_size (int): Size of the kernel. Defaults to 3.
        - bias (bool): If True, adds a learnable bias to the output. Defaults to True.
        - dilation (int): Spacing between kernel elements. Defaults to 1.
        
        - nonlinearity (Optional[str], optional): Non-linear function to apply. Defaults to "relu".
        - normalization (Optional[str], optional): Normalization to apply. Defaults to "bn".

        - model_depth (int): Depth of the encoding path. Defaults to 4.
        - root_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 16.
        - num_feat_maps (int): Base multiplier for output channels numberfor multiplication. Defaults to 16.
        - num_conv_blocks (int): Number of convolutions per block at specific depth. Defaults to 2.   
        - final_activation (str): Name of the final activation to use. Defaults to sigmoid.

    Raises:
        ValueError: Invalid normalization value
        ValueError: Invalid nonlinearity value
    """

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
                final_activation: str="sigmoid"):
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
                                    final_activation=final_activation,
                                    group=group,
                                    group_dim=group_dim)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        self.logger.debug(f"Final output shape: {x.shape}")
        return x
