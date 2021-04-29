"""Models for the RFNN experiments"""
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf
import torch


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
        # torch.nn.init.kaiming_normal_(w, nonlinearity="relu")
        torch.nn.init.constant_(w, 2.0) # For testing purposes
        return w


    def conv(self, x, kernel_size, n_out, strides=1, padding="same"):
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
        n_in = x.shape[1] # number of input channels
        W = self.get_kernel([n_out, n_in, kernel_size, kernel_size, kernel_size])

        # TODO put padding = padding when 1.9.1 comes out
        return torch.nn.functional.conv3d(x, W, stride=(strides,strides,strides), padding=1)


    def conv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
                   padding="same", fnc=torch.nn.functional.relu):
        """Convolution with batch normalization/bias and nonlinearity"""
        y = self.conv(x, kernel_size, n_out, strides=strides, padding=padding)
        if use_bn:
            # TODO verify that this is sufficiently close to TF2 (weights ?)
            BatchNormalization = torch.nn.BatchNorm3d(n_out)
            return fnc(BatchNormalization(y))
        else:
            bias = torch.nn.init.constant_(torch.empty(list(y.shape)), 0.01)
            return fnc(torch.add(y, bias))


    def Gconv_block(self, x, kernel_size, n_out, is_training, use_bn=True, strides=1,
            padding="same", fnc=torch.nn.functional.relu, name="Gconv_block", drop_sigma=0.1):
        """Convolution with batch normalization/bias and nonlinearity"""
        y = self.Gconv(x, kernel_size, n_out, is_training, strides=strides, padding=padding, drop_sigma=drop_sigma)
        y = y.permute([0,1,2,3,5,4]) # needs to be recheked
        ysh = y.shape
        if use_bn:
            y = fnc(torch.nn.functional.batch_norm(y, torch.zeros(n_out), torch.ones(n_out)))
        else:
            bias = torch.nn.init.constant_(torch.empty(list(y.shape)), 0.01) 
            y = fnc(torch.add(y, bias))
        return y.permute([0,1,2,3,5,4])


    def Gconv(self, x, kernel_size, n_out, is_training, strides=1, padding="same", drop_sigma=0.1):
        """Perform a discretized convolution on SO(3)

        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
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

        # print(f"\n\nxsh : {xsh}")
        # W is the base filter. We rotate it 4 times for a p4 convolution over
        # R^2. For a p4 convolution over p4, we rotate it, and then shift up by
        # one dimension in the channels.
        W = self.get_kernel([n_in* xsh[5]* n_out, kernel_size, kernel_size, kernel_size])
        # print(f"W : {W.shape}")
        
        WN = self.group.get_Grotations(W)
        # print(f"WN : {WN[0].shape}")

        WN = torch.stack(WN, -1)
        # print(f"WN stacked: {WN.shape}")

        # Reshape and rotate the io filters 4 times. Each input-output pair is
        # rotated and stacked into a much bigger kernel
        # print(f"x: {x.shape}")
        xN = torch.reshape(x, [batch_size, n_in*xsh[5], xsh[2], xsh[3], xsh[4]])
        # print(f"xN: {xN.shape}")

        if xsh[5] == 1:
            # A convolution on R^2 is just standard convolution with 3 extra 
            # output channels for each rotation of the filters
            WN = torch.reshape(WN, [-1, n_in, kernel_size, kernel_size, kernel_size])
            # print(f"WN stacked, if: {WN.shape}")

        elif xsh[5] == self.group_dim:
            # A convolution on p4 is different to convolution on R^2. For each
            # dimension of the group output, we need to both rotate the filters
            # and circularly shift them in the input-group dimension. In a
            # sense, we have to spiral the filters
            WN = torch.reshape(WN, [n_in, kernel_size, kernel_size, kernel_size, self.group_dim, n_out, self.group_dim])
            # [kernel_size, kernel_size, kernel_size, n_in, 4, n_out, 4]
            # Shift over axis 4
            # print(f"WN stacked, before permut: {WN.shape}")

            WN_shifted = self.group.G_permutation(WN)
            WN = torch.stack(WN_shifted, -1)
            # print(f"WN stacked, after permut: {WN.shape}")

            # Shift over axis 6
            # Stack the shifted tensors and reshape to 4D kernel
            WN = torch.reshape(WN, [n_out*self.group_dim, n_in*self.group_dim, kernel_size, kernel_size, kernel_size])
            # [kernel_size, kernel_size, kernel_size, xsh[4]*self.group_dim, n_out*self.group_dim]
            # print(f"WN final: {WN.shape}")

        # Convolve
        # Gaussian dropout on the weights
        WN *= (1 + drop_sigma*float(is_training)*torch.randn(WN.shape))

        if not (isinstance(strides, tuple) or isinstance(strides, list)):
            strides = (strides,strides,strides)
        if padding == 'reflect':
            padding = 'valid'
            pad = WN.shape[2] // 2
            xN = tf.pad(tensor=xN, paddings=[[0,0],[pad,pad],[pad,pad],[pad,pad],[0,0]], mode='REFLECT') 

        # TODO put padding=padding in 1.9.0, verify strides (before strides = strides)
        yN = torch.nn.functional.conv3d(xN, WN, stride=strides, padding=1)
        ysh = yN.shape
        # print(f"ysh: {ysh}\n\n")

        y = torch.reshape(yN, [batch_size, n_out, ysh[2], ysh[3], ysh[4], self.group_dim]) # Dims to check
        return (y, WN)


    def Gres_block(self, x, kernel_size, n_out, is_training, use_bn=True,
                   strides=1, padding="same", fnc=tf.nn.relu, drop_sigma=0.1,  name="Gres_block"):
        """Residual block style 3D group convolution
        
        Args:
            x: [batch_size, height, width, n_in, group_dim/1]
            kernel_size: int for the spatial size of the kernel
            n_out: int for number of output channels
            strides: int for spatial stride length
            padding: "valid" or "same" padding
        Returns:
            [batch_size, new_height, new_width, new_depth, n_out, group_dim] tensor in G
        """
        # Begin residual connection
        y = self.Gconv_block(x, kernel_size, n_out, is_training, use_bn=use_bn, strides=strides, 
                                padding=padding, fnc=fnc, drop_sigma=drop_sigma, name="Gconv_blocka")
        y = self.Gconv_block(y, kernel_size, n_out, is_training, use_bn=use_bn, drop_sigma=drop_sigma,
                                fnc=tf.identity, name="Gconv_blockb")

        # Recombine with shortcut
        # a) resize and pad input if necessary
        xsh = tf.shape(input=x)
        ysh = tf.shape(input=y)
        xksize = (1,kernel_size,kernel_size,kernel_size,1)
        xstrides = (1,strides,strides,strides,1)
        x = tf.reshape(x, tf.concat([xsh[:4],[-1,]], 0))
        x = tf.nn.avg_pool3d(x, xksize, xstrides, "same")
        x = tf.reshape(x, tf.concat([ysh[:4],[-1,self.group_dim]], 0))
        
        diff = n_out - x.get_shape().as_list()[-2]
        paddings = tf.constant([[0,0],[0,0],[0,0],[0,0],[0,diff],[0,0]])
        x = tf.pad(tensor=x, paddings=paddings)
        
        # b) recombine
        #return fnc(x+y)
        return x+y
