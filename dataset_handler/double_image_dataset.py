import torch
import torchvision
from skimage import color


class DoubleImageDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        x, _ = super(DoubleImageDataset, self).__getitem__(index)
        print(x.shape)
        x_A = x[:, :0.5 * x.shape[1], :]
        x_B = x[:, 0.5 * x.shape[1]:, :]
        return x_A, x_B

