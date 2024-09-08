import torch.nn as nn

from utils import bchw_2_blc


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[7, 3], patch_size=2):
        super(PatchEmbed, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size_1, kernel_size_2 = kernel_size
        elif isinstance(kernel_size, list):
            kernel_size_1, kernel_size_2 = kernel_size[0], kernel_size[1]
        else:
            ValueError("kernel_size must be an integer or a list")

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_1,
                      stride=patch_size, padding=(kernel_size_1-patch_size+1)//2, groups=in_channels, padding_mode='reflect'),
            nn.LeakyReLU(True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=kernel_size_2, stride=1, padding=1, groups=out_channels, padding_mode='reflect'),
            nn.LeakyReLU(True),
        )
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        return self.norm(bchw_2_blc(self.layer(x)))
