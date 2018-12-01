import torch
import torchvision


class ColorizationDataset(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        x, _ = super(ColorizationDataset, self).__getitem__(index)
        bw_x = torch.mean(x, dim=0).unsqueeze(0)
        return torch.cat([bw_x] * 3, dim=0), x
        
