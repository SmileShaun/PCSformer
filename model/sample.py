import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size=None):
        super(DownSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = patch_size if kernel_size is None else kernel_size

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=patch_size, padding=(
            kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, kernel_size=None):
        super(UpSample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = 1 if kernel_size is None else kernel_size

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.layer(x)