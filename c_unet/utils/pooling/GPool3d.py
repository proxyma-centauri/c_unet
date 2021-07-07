from einops import reduce
from torch import nn


class GPool3d(nn.Module):
    """Pool over channels using einops.
    
    pool_over: `c`, `g`, `cg` or `hwd`
    reduction: `max` or `mean`
    """
    def __init__(self,
                 pool_over: str = "c",
                 reduction: str = "mean",
                 reduction_factor: int = 2):
        super(GPool3d, self).__init__()
        self.pool_over = pool_over
        self.reduction = reduction
        self.reduction_factor = reduction_factor

    def forward(self, x):
        if self.pool_over == "c":
            return reduce(x,
                          "b (c c2) g h w d -> b c g h w d",
                          reduction=self.reduction,
                          c2=self.reduction_factor)
        elif self.pool_over == "g":
            return reduce(x,
                          "b c (g g2) h w d -> b c g h w d",
                          reduction=self.reduction,
                          g2=self.reduction_factor)
        elif self.pool_over == "cg":
            return reduce(x,
                          "b (c c2) (g g2) h w d -> b c g h w d",
                          reduction=self.reduction,
                          c2=self.reduction_factor,
                          g2=self.reduction_factor)
        elif self.pool_over == "hwd":
            return reduce(x,
                          "b c g (h h2) (w w2) (d d2) -> b c g h w d",
                          reduction=self.reduction,
                          h2=self.reduction_factor,
                          w2=self.reduction_factor,
                          d2=self.reduction_factor)
