import torch
from torch import nn


class ReshapedCat(nn.Module):
    """ Performs concatenation of the decoder path,
        after having reshaped the data into 5d Tensors;
        This helps in stabilizing the network."""
    def __init__(self):
        super(ReshapedCat, self).__init__()

    def forward(self, x, down_sampling_feature):
        # Reshaping
        bs, c, g, h, w, d = down_sampling_feature.shape
        down_sampling_feature = down_sampling_feature.reshape(
            [bs, c * g, h, w, d])

        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])

        # Concatenating
        x = torch.cat((down_sampling_feature, x), dim=1)

        # Reshaping
        bs, c, h, w, d = x.shape
        c_cat = c // g
        x = x.reshape([bs, c_cat, g, h, w, d])
        return x
