import torch
import torchvision
from skimage import color

from .abstract_dataset import AbstractDataset


class ColorizationDataset(AbstractDataset):

    def __getitem__(self, index):
        x, _ = super(ColorizationDataset, self).__getitem__(index)
        numpy_x = x.numpy()
        swap_x = x.numpy().transpose(1, 2, 0)
        lab_x = color.rgb2lab(swap_x).transpose(2, 0, 1)
        x = 1. / 100 * torch.from_numpy(lab_x).type(torch.FloatTensor)
        bw_x = x[0:1, :, :]
        return bw_x, x

