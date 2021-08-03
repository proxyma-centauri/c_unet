import torch
from torch import nn


class OperationAndCat(nn.Module):
    """ Performs concatenation of the decoder path,
        after having reshaped the data into 5d Tensors;
        This helps in stabilizing the network."""
    def __init__(self, logger):
        super(OperationAndCat, self).__init__()
        self.logger = logger

    def forward(self, x, down_sampling_feature, key, layer, group):

        # Reshaping if needed
        if group:
            bs, c, g, h, w, d = x.shape
            x = x.reshape([bs, c * g, h, w, d])

            bs, c, g, h, w, d = down_sampling_feature.shape
            down_sampling_feature = down_sampling_feature.reshape(
                [bs, c * g, h, w, d])

        # Upsampling
        x = layer(x)
        self.logger.debug(f"{key}, {x.shape}")

        # Concatenating
        x = torch.cat((down_sampling_feature, x), dim=1)

        # Reshaping
        if group:
            bs, c, h, w, d = x.shape
            c_cat = c // g
            x = x.reshape([bs, c_cat, g, h, w, d])

        return x
