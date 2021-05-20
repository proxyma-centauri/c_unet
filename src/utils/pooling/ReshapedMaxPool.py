from torch import nn
from typing import List, Union


class ReshapedMaxPool(nn.Module):
    """ Performs AveragePooling through MaxPool3d,
        after having reshaped the data into a 5d Tensor"""

    def __init__(self,
                kernel_size: int = 3,
                stride: Union[int, List[int]] = 1,
                padding: Union[str, int] = 1):

        super(ReshapedMaxPool, self).__init__()
        self.MaxPool = nn.MaxPool3d(kernel_size, 
                    stride, 
                    padding
        )

    def forward(self, x):
        bs, c, g, h, w, d = x.shape
        x = x.reshape([bs, c * g, h, w, d])
        x = (self.MaxPool(x)).view([bs, c, g, h, w, d])
        return x