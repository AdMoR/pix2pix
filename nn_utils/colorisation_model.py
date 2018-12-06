import torch

from .model import UNet


class ColorUNet(UNet):

    def __init__(self):
        super(ColorUNet, self).__init__(layers=[1, 64, 128, 256, 512, 512, 512, 512, 512], target_dim=2)

    def forward(self, x, z=None):
        ab_channel = super(ColorUNet, self).forward(x, z)
        return torch.cat([x, ab_channel], dim=1)

