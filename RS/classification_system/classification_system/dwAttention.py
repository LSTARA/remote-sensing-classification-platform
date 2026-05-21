import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        mid_channels = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class GGAM(nn.Module):
    def __init__(self, gate_channels, spatial_size=None, reduction_ratio=16,
                 groups=64, sa_kernel_size=3, is_sigmoid=True, no_spatial=False):
        super(GGAM, self).__init__()
        self.attention = ChannelAttention(gate_channels, reduction_ratio)

    def forward(self, x):
        weight = self.attention(x)
        return x * weight, weight