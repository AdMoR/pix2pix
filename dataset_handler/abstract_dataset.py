import torchvision
import torch
from skimage import color
import torchvision.utils as vutils


class AbstractDataset(torchvision.datasets.ImageFolder):

    @classmethod
    def lab_to_rgb(cls, x):
        for i in range(x.shape[0]):
            swap_x = 100 * x[i, :, :, :].cpu().detach().numpy().transpose(1, 2, 0)
            x[i, :, :, :] = torch.from_numpy(color.lab2rgb(swap_x).transpose(2, 0, 1))
        return x

    @classmethod
    def build_visu(cls, writer, gen, x, y, device, index):
        gray_scale = torch.cat([x for _ in range(3)], dim=1)
        viz = vutils.make_grid(torch.cat([cls.lab_to_rgb(y).to(device), cls.lab_to_rgb(gen(x.to(device))),  gray_scale.to(device)], dim=0))
        viz = torch.clamp(viz, -0.9999999, 0.999999)
        writer.add_image('visu/', viz, index)
