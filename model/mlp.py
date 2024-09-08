import torch.nn as nn
from timm.models.layers import trunc_normal_
from torch.nn.init import _calculate_fan_in_and_fan_out
import math


class Mlp(nn.Module):
    def __init__(self, in_dim, network_depth, hidden_dim=None, out_dim=None):
        super(Mlp, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, out_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
