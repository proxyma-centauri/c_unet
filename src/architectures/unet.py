import torch.nn as nn
from src.layers.gconvs import GconvResBlock, GconvBlock
from src.layers.convs import ConvBlock

class Unet(nn.Module):

    def __init__(self,
                group: str,
                group_dim: int,
                in_channels: int,
                internal_channels: int,
                out_channels: int,
                kernel_size: int = 3,
                bias: bool = True,):
        super(Unet, self).__init__()

        self.seq = nn.Sequential(
           GconvBlock(group, 1, in_channels, internal_channels, dropout=0, normalization="sn"),
           GconvBlock(group, group_dim, internal_channels, internal_channels, dropout=0, normalization="sn"),
           GconvResBlock(group, group_dim, internal_channels, internal_channels, dropout=0, normalization="sn")
        )

        # Number of channels must be equal to number of classes in target
        self.finalConv = ConvBlock(internal_channels*group_dim, out_channels, nonlinearity="", normalization="sn")

    def forward(self, x):
        x = self.seq(x)

        # Reshaping x to right shape
        bs, c, g, h, w, d = x.shape
        x = x.reshape(bs, c*g, h, w, d)
        x = self.finalConv(x)

        return x