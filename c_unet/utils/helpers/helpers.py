from torch import nn
from typing import List, Union


def calc_same_padding(input_,
                      kernel=1,
                      stride=1,
                      dilation=1,
                      transposed=False) -> int:
    """
    Helper function to calculate the padding to add to conserve shaped after convolutions

    Args:
        - input_ (torch tensor): input tensor
        - kernel (int): kernel of the convolution
        - stride (int): stride of the convolution
        - dilation (int): dilation of the convolution
        - transposed (bool): Whether or not the convolution is a transposed one
    """
    if transposed:
        return (dilation * (kernel - 1) + 1) // 2 - 1, input_ // (1. / stride)
    else:
        return (dilation * (kernel - 1) + 1) // 2, input_ // stride


def calc_upsampling_size(input_size: int,
                         dilation: int = 1,
                         tconv_kernel_size: int = 3,
                         tconv_stride: int = 2,
                         tconv_padding: int = 1,
                         output_padding: int = 1) -> int:
    """
    Helper function to calculate the upsampling size desired from the convolution parameters.
    This is the same formula as used by the transposed 3D convolution of PyTorch

    Args:
        - input_size (int): size of the input tensor
        - dilation (int): dilation of the convolution
        - tconv_kernel_size (int): kernel of the convolution
        - tconv_stride (int): stride of the convolution
        - tconv_padding (int): padding of the convolution
        - output_padding (int): Output padding to add
    """
    return (input_size - 1) * tconv_stride - 2 * tconv_padding + dilation * (
        tconv_kernel_size - 1) + output_padding + 1


def conv3d(in_channels: int,
           out_channels: int,
           kernel_size: int = 3,
           stride: Union[int, List[int]] = 1,
           padding: Union[str, int] = 1,
           bias: bool = True,
           dilation: int = 1) -> nn.Module:
    """Applies a 3D convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel. Defaults to 3.
        stride (Union[int, List[int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[str, int], optional): Zero-padding added to all three sides of the input. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.

    Raises:
        ValueError: Invalid padding value

    Returns:
        nn.Module: Conv3d
    """

    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     dilation=dilation)
