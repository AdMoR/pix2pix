import os
import random
import collections
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data
import torchvision.utils as vutils

from .abstract_dataset import AbstractDataset



def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]



class ADE20KLoader(AbstractDataset):

    n_classes = 150

    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)

        for split in ["training", "validation"]:
            file_list = recursive_glob(
                rootdir=self.root + "images/" + self.split + "/", suffix=".jpg"
            )
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path[:-4] + "_seg.png"

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        #img -= self.mean
        #if self.img_norm:
        # Resize scales images from 0 to 255, thus we need
        # to divide by 255.0
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        lbl = self.encode_segmap(lbl)
        #classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        #assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]))
        label_mask = (mask[:, :, 0] / 10.0) * 256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.uint8)

    @classmethod
    def decode_segmap(cls, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, cls.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


class EdgeADE20k(ADE20KLoader):

    @classmethod
    def lab_to_rgb(cls, x):
        return x

    def __init__(self, *args, **kwargs):
        super(EdgeADE20k, self).__init__(*args, **kwargs)
        filters = torch.cat(
            [torch.from_numpy(np.array([[-1, 0, 1], [0, 1, 0], [0, -1, 0]])).unsqueeze(0).unsqueeze(0)],
            dim=0)
        self.weight = filters.float()

    @classmethod
    def build_edge_img(cls, lbl, weight):
        lbl = torch.abs(F.conv2d(lbl.float().unsqueeze(0).unsqueeze(0), weight, padding=1, groups=1))
        lbl[lbl > 0] = 1
        return lbl.squeeze(0)

    def __getitem__(self, index):
        try:
            img, lbl = super(EdgeADE20k, self).__getitem__(index)
            lbl = self.build_edge_img(lbl, self.weight)
            return lbl, img
        except:
            return self.__getitem__(random.randint(0, self.__len__()))


class NormalADE20k(EdgeADE20k):

    @classmethod
    def build_x_img(cls, x):
        B, C, H, W = x.shape
        final = list()
        for b in range(B):
            out = torch.zeros(H, W)
            for l in range(cls.n_classes):
                out += l * (x[b, l, :, :] == 1).float()
            img = torch.from_numpy(cls.decode_segmap(out.long().numpy())).unsqueeze(0)
            img = img.permute(0, 3, 1, 2)
            final.append(img)
        out = torch.cat(final, dim=0).float()
        return out

    def __getitem__(self, index):
        try:
            img, lbl = super(EdgeADE20k, self).__getitem__(index)
            new_lbl = torch.zeros((self.n_classes + 1,) + self.img_size)
            for l in range(self.n_classes):
                new_lbl[l, :, :] = (lbl == l)
            new_lbl[150, :, :] = self.build_edge_img(lbl, self.weight)
            return new_lbl, img
        except:
            return self.__getitem__(random.randint(0, self.__len__()))



if __name__ == "__main__":
    local_path = "/Users/meet/data/ADE20K_2016_07_26/"
    dst = ADE20KLoader(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
plt.show()
