"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def calc_same_padding(
    input_,
    kernel=1,
    stride=1,
    dilation=1,
    transposed=False,
):
    if transposed:
        return (dilation * (kernel - 1) + 1) // 2 - 1, input_ // (1. / stride)
    else:
        return (dilation * (kernel - 1) + 1) // 2, input_ // stride


class Layers(object):

    def __init__(self, group):
        if group == "V":
            from V_group import V_group
            self.group = V_group()
            self.group_dim = self.group.group_dim
        elif group == "S4":
            from S4_group import S4_group
            self.group = S4_group()
            self.group_dim = self.group.group_dim
        elif group == "T4":
            from T4_group import T4_group
            self.group = T4_group()
            self.group_dim = self.group.group_dim
        else:
            print("Group is not recognized")
            sys.exit(-1)

        # Constants
        self.cayley = self.group.cayleytable

    def get_kernel(self, shape, trainable=True):
        w = torch.empty(shape, requires_grad=trainable)
        # nn.init.kaiming_normal_(w, nonlinearity="relu")
        nn.init.constant_(w, 2.0)  # For testing purposes
        return w

    def conv(self,
             x,
             kernel_size,
             n_out,
             strides=1,
             padding=1,
             input_size=None):
        """A basic 3D convolution

        Args:
            x: [batch_size, n_in, height, width, depth]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, n_out, new_height, new_width, new_depth, group_dim] tensor in G
        """
        n_in = x.shape[1]  # number of input channels
        W = self.get_kernel(
            [n_out, n_in, kernel_size, kernel_size, kernel_size])

        # TODO put padding = padding when 1.9.1 comes out
        if padding == "same":
            assert input_size, "`padding='same'` requires the argument `input_size` before torch 1.9.1"
            p, _ = calc_same_padding(input_=input_size,
                                     kernel=kernel_size,
                                     stride=strides,
                                     transposed=False)
        elif type(padding) == int:
            p = padding
        else:
            raise ValueError(f"Invalid padding value: {padding}")

        return F.conv3d(x, W, stride=(strides, strides, strides), padding=p)

    def conv_block(self,
                   x,
                   kernel_size,
                   n_out,
                   is_training,
                   use_bn=True,
                   strides=1,
                   padding="same",
                   fnc=F.relu):
        """Convolution with batch normalization/bias and nonlinearity"""
        y = self.conv(x, kernel_size, n_out, strides=strides, padding=padding)
        if use_bn:
            # TODO verify that this is sufficiently close to TF2 (weights ?)
            BatchNormalization = nn.BatchNorm3d(n_out)
            return fnc(BatchNormalization(y))
        else:
            bias = nn.init.constant_(torch.empty(list(y.shape)), 0.01)
            return fnc(torch.add(y, bias))

    def Gconv_block(self,
                    x,
                    kernel_size,
                    n_out,
                    is_training,
                    use_bn=True,
                    strides=1,
                    padding="same",
                    fnc=F.relu,
                    name="Gconv_block",
                    drop_sigma=0.1):
        """Convolution with batch normalization/bias and nonlinearity"""
        y = self.Gconv(x,
                       kernel_size,
                       n_out,
                       is_training,
                       strides=strides,
                       padding=padding,
                       drop_sigma=drop_sigma)
        y = y.permute([0, 2, 1, 3, 4, 5])
        ysh = y.shape
        if use_bn:
            y = torch.reshape(
                y, [ysh[0], n_out * self.group_dim, ysh[3], ysh[4], ysh[5]])
            BatchNormalization = nn.BatchNorm3d(n_out * self.group_dim)
            y = fnc(torch.reshape(BatchNormalization(y), ysh))
        else:
            bias = nn.init.constant_(torch.empty(list(y.shape)), 0.01)
            y = fnc(torch.add(y, bias))
        return y.permute([0, 2, 1, 3, 4, 5])

    def Gconv(self,
              x,
              kernel_size,
              n_out,
              is_training,
              strides=1,
              padding="same",
              drop_sigma=0.1):
        """Perform a discretized convolution on SO(3)

        Args:
            x: [batch_size, n_in, group_dim|1, height, width, depth]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        xsh = x.shape

        batch_size = xsh[0]
        n_in = xsh[1]

        # W is the base filter. We rotate it 4 times for a p4 convolution over
        # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
        # one dimension in the channels.
        W = self.get_kernel(
            [n_in * xsh[2] * n_out, kernel_size, kernel_size, kernel_size])
        WN = self.group.get_Grotations(W)
        WN = torch.stack(WN, 0)

        # Reshape and rotate the io filters 4 times. Each input-output pair is
        # rotated and stacked into a much bigger kernel
        xN = torch.reshape(x,
                           [batch_size, n_in * xsh[2], xsh[3], xsh[4], xsh[5]])

        if xsh[2] == 1:
            # A convolution on R^2 is just standard convolution with 3 extra
            # output channels for each rotation of the filters
            WN = torch.reshape(
                WN, [-1, n_in, kernel_size, kernel_size, kernel_size])

        elif xsh[2] == self.group_dim:
            # A convolution on p4 is different to convolution on R^2. For each
            # dimension of the group output, we need to both rotate the filters
            # and circularly shift them in the input-group dimension. In a
            # sense, we have to spiral the filters
            WN = torch.reshape(WN, [
                n_in, kernel_size, kernel_size, kernel_size, self.group_dim,
                n_out, self.group_dim
            ])
            # [kernel_size, kernel_size, kernel_size, n_in, 4, n_out, 4]
            # Shift over axis 4

            WN_shifted = self.group.G_permutation(WN)
            WN = torch.stack(WN_shifted, -1)

            # Shift over axis 6
            # Stack the shifted tensors and reshape to 4D kernel
            WN = torch.reshape(WN, [
                n_out * self.group_dim, n_in * self.group_dim, kernel_size,
                kernel_size, kernel_size
            ])
            # [kernel_size, kernel_size, kernel_size, xsh[4]*self.group_dim, n_out*self.group_dim]

        # Convolve
        # Gaussian dropout on the weights
        WN *= (1 + drop_sigma * float(is_training) * torch.randn(WN.shape))

        if not (isinstance(strides, tuple) or isinstance(strides, list)):
            strides = (strides, strides, strides)
        if padding == 'reflect':
            padding = 'valid'
            pad = WN.shape[2] // 2
            xN = tf.pad(tensor=xN,
                        paddings=[[0, 0], [pad, pad], [pad, pad], [pad, pad],
                                  [0, 0]],
                        mode='REFLECT')

        # TODO put padding=padding in 1.9.0, verify strides (before strides = strides)
        yN = F.conv3d(xN, WN, stride=strides, padding=1)
        ysh = yN.shape
        y = torch.reshape(
            yN, [batch_size, n_out, self.group_dim, ysh[2], ysh[3], ysh[4]])
        return (y)

    def Gres_block(self,
                   x,
                   kernel_size,
                   n_out,
                   is_training,
                   use_bn=True,
                   strides=1,
                   padding="same",
                   fnc=F.relu,
                   drop_sigma=0.1,
                   name="Gres_block"):
        """Residual block style 3D group convolution
        
        Args:
            x: [batch_size, n_in, group_dim|1, height, width, depth]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        # Begin residual connection
        y = self.Gconv_block(x,
                             kernel_size,
                             n_out,
                             is_training,
                             use_bn=use_bn,
                             strides=strides,
                             padding=padding,
                             fnc=fnc,
                             drop_sigma=drop_sigma,
                             name="Gconv_blocka")
        y = self.Gconv_block(y,
                             kernel_size,
                             n_out,
                             is_training,
                             use_bn=use_bn,
                             drop_sigma=drop_sigma,
                             fnc=nn.Identity(),
                             name="Gconv_blockb")

        # Recombine with shortcut
        # a) resize and pad input if necessary
        xsh = x.shape
        ysh = y.shape
        xksize = (1, kernel_size, kernel_size, kernel_size, 1)
        xstrides = (1, strides, strides, strides, 1)

        x = torch.reshape(x, tf.concat([xsh[:4], [
            -1,
        ]], 0))
        x = tf.nn.avg_pool3d(x, xksize, xstrides, "same")
        x = tf.reshape(x, tf.concat([ysh[:4], [-1, self.group_dim]], 0))

        diff = n_out - xsh[-2]
        paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 0], [0, diff],
                                [0, 0]])
        x = tf.pad(tensor=x, paddings=paddings)

        # b) recombine
        #return fnc(x+y)
        return x + y


# [BEGIN REFACTORING]


def conv3d(in_channels: int,
           out_channels: int,
           kernel_size: int = 3,
           stride: Union[int, List[int]] = 1,
           padding: Union[str, int] = 1,
           bias: bool = True,
           dilation: int = 1) -> nn.Module:
    """[summary]

    Args:
        in_channels (int): [description]
        out_channels (int): [description]
        kernel_size (int): [description]. Defaults to 3.
        stride (Union[int, List[int]], optional): [description]. Defaults to 1.
        padding (Union[str, int], optional): [description]. Defaults to 1.
        bias (bool, optional): [description]. Defaults to True.
        dilation (int, optional): [description]. Defaults to 1.

    Raises:
        ValueError: [description]

    Returns:
        nn.Module: [description]
    """
    if padding == "same":
        p = (dilation * (kernel_size - 1) + 1) // 2
    elif isinstance(padding, int):
        p = padding
    else:
        raise ValueError(f"Invalid padding value: {padding}")

    return nn.Conv3d(in_channels,
                     out_channels,
                     kernel_size,
                     stride=stride,
                     padding=p,
                     bias=bias,
                     dilation=dilation)


class ConvBlock(nn.Module):
    """[summary]

    Args:
        in_channels (int): [description]
        out_channels (int): [description]
        kernel_size (int): [description]. Defaults to 3.
        stride (Union[int, List[int]], optional): [description]. Defaults to 1.
        padding (Union[str, int], optional): [description]. Defaults to 1.
        bias (bool, optional): [description]. Defaults to True.
        dilation (int, optional): [description]. Defaults to 1.
        nonlinearity (Optional[str], optional): [description]. Defaults to "relu".
        normalization (Optional[str], optional): [description]. Defaults to "bn".

    Raises:
        ValueError: [description]
        ValueError: [description]
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: Union[int, List[int]] = 1,
                 padding: Union[str, int] = 1,
                 bias: bool = True,
                 dilation: int = 1,
                 nonlinearity: Optional[str] = "relu",
                 normalization: Optional[str] = "bn"):
        super(ConvBlock, self).__init__()

        modules = [
            conv3d(in_channels, out_channels, kernel_size, stride, padding,
                   bias, dilation)
        ]

        # ! WARNING: You'll end up using a batch size of 1, we need another
        # ! normalization layer (e.g. switchnorm).
        if normalization:
            if normalization == "bn":
                modules.append(nn.BatchNorm3d(out_channels))
            else:
                raise ValueError(
                    f"Invalid normalization value: {normalization}")

        if nonlinearity:
            if nonlinearity == "relu":
                modules.append(nn.ReLU(inplace=True))
            else:
                raise ValueError(f"Invalid nonlinearity value: {nonlinearity}")

        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class GaussianDropout(nn.Module):
    """[summary]

    Args:
        α (float): [description]. Defaults to 1.0.
    """

    def __init__(self, α: float = 1.0):
        super(GaussianDropout, self).__init__()
        self.α = torch.Tensor([α])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, α)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, α)
            ε = torch.randn(x.size()) * self.α + 1

            ε = Variable(ε)

            return x * ε
        else:
            return x
