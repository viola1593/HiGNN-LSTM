from torch import nn

'''Adapted from: https://github.com/corentin-dfg/graph4sen2dwater/blob/main/models/graphcast/mlp.py'''

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, last=False):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Hardswish(), 
            nn.Linear(hid_dim, out_dim),
            nn.LayerNorm(out_dim) if not last else nn.Identity()
        )

    def forward(self, x):
        return self.mlp(x)