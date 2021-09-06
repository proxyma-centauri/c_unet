from torch import nn


class ReshapedBatchNorm(nn.Module):
    """
    Performs BatchNormalization through BatchNorm3d,
    after having reshaped the data into a 5d Tensor
    
    Args:
        - in_channels (int): number of input channels
        - group_dim (int): dimension of the group
    """
    def __init__(self, in_channels: int, group_dim: int):
        super(ReshapedBatchNorm, self).__init__()
        self.BatchNormalization = nn.BatchNorm3d(in_channels * group_dim)

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = (self.BatchNormalization(x)).view([bs, c, g, h, w, d])
        return x
