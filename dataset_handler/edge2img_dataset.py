import torch
import torchvision
import cv2
import numpy as np
from skimage import color


class EdgesDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        x, _ = super(EdgesDataset, self).__getitem__(index)
        swap_x = x.numpy().transpose(1, 2, 0)
        mult_swap_x = 255 * swap_x

        canny_x = cv2.Canny(np.uint8(mult_swap_x), 50, 150)

        edges_x = torch.from_numpy(canny_x).unsqueeze(0).type(torch.FloatTensor)
        edges_x = 1./255 * edges_x


        lab_x = color.rgb2lab(swap_x).transpose(2, 0, 1)
        x = 1. / 100 * torch.from_numpy(lab_x).type(torch.FloatTensor)

        return edges_x, x

