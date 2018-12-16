import torch
import torchvision


class PerceptualModel(torch.nn.Module):

    def __init__(self, model=torchvision.models.vgg16):
        super(PerceptualModel, self).__init__()
        features = model(pretrained=True).features
        blocks = [0] + [i for i in range(len(features)) if features[i].__class__.__name__ == "MaxPool2d" ]
        self.model_blocks = [torch.nn.Sequential(features[blocks[i]: blocks[i + 1]]) for i in range(len(blocks) - 1)]

    def forward(self, x, *args, **kwargs):
        outs = list()
        for block in self.model_blocks:
            x = block(x)
            outs.append(x)
        return outs


class PerceptualLoss(torch.nn.Module):

    def __init__(self, model=None, device=torch.device("cpu")):
        super(PerceptualLoss, self).__init__()
        if model is None:
            self.model = PerceptualModel()
        elif model.__class__.__name__ == "function":
            self.model = PerceptualModel(model)
        else:
            self.model = model
        self.device = device

    def forward(self, x_1, x_2):
        if type(x_1) != list:
            x_1, x_2 = self.model(x_1), self.model(x_2)
        return (1. / len(x_1)) * torch.sum(torch.cat(
             [torch.norm(x_1[i] - x_2[i], 1)
             for i in range(len(x_1))])
        )
