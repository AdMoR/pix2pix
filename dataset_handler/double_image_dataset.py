import torch
import torchvision
from skimage import color


class DoubleImageDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        x, _ = super(DoubleImageDataset, self).__getitem__(index)
        x_A = x[:, :, :int(0.5 * x.shape[2])]
        x_B = x[:, :, int(0.5 * x.shape[2]):]
        return x_A, x_B

