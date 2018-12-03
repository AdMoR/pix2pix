import torch
import torch.nn as nn
from torchvision import models


class BasicEncoderBlock(nn.Module):

    def __init__(self, in_, out_, act_fn=nn.ReLU, pool_fn=nn.MaxPool2d):
        super(BasicEncoderBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_, in_, 3, padding=1),
            nn.BatchNorm2d(in_),
            act_fn(inplace=True),
            nn.Conv2d(in_, out_, 3, padding=1),
            nn.BatchNorm2d(out_),
            act_fn(inplace=True)
        )
        if pool_fn is not None:
            self.pool = pool_fn(kernel_size=(2, 2), stride=2)
        else:
            self.pool = None

    def forward(self, x):
        if self.pool is not None:
            return self.pool(self.op(x))
        else:
            return self.op(x)


class BasicDecoderBlock(torch.nn.Module):

    def __init__(self, in_, out_, act_fn):
        super(BasicDecoderBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_, in_, 3, padding=1),
            nn.BatchNorm2d(in_),
            act_fn(inplace=True),
            nn.Conv2d(in_, out_, 3, padding=1),
            nn.BatchNorm2d(out_),
            act_fn(inplace=True)
        )

    def forward(self, x, previous):
        x_prime = nn.functional.interpolate(x, scale_factor=(2, 2))
        combined = torch.cat([x_prime, previous], dim=1)
        return self.op(combined)


class UNet(torch.nn.Module):

    def __init__(self, layers):
        super(UNet, self).__init__()
        self.act_fn = nn.ReLU
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            self.encoder[str(i)] = BasicEncoderBlock(in_, out_, self.act_fn)
        for i, (in_, out_) in enumerate(zip(layers[1:], layers[:-1])):
            self.decoder[str(i)] = BasicDecoderBlock(in_ + out_, out_, self.act_fn)

    def forward(self, x, z=None):
        inputs = dict()
        for i in range(len(self.encoder)):
            inputs[i] = x
            x = self.encoder[str(i)](x)

        for i in range(len(self.encoder)):
            x = self.decoder[str(len(self.encoder) - (i + 1))](x, inputs[len(self.encoder) - (i + 1)])
        return torch.sigmoid(x)


class EncoderNet(nn.Module):

    def __init__(self, layers, n_classes=1):
        super(EncoderNet, self).__init__()
        self.act_fn = nn.ReLU
        self.encoder = nn.ModuleDict()

        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            self.encoder[str(i)] = BasicEncoderBlock(in_, out_, self.act_fn)
        self.linear = nn.Linear(layers[-1], n_classes)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, y=None):
        if y is not None:
            z = torch.cat([x, y], dim=1)
        else:
            z = x
        for i in range(len(self.encoder)):
            z = self.encoder[str(i)](z)
        pooled_z = self.pool(z).view(x.shape[0], -1)
        return torch.sigmoid(self.linear(pooled_z))


class AlexNet_finetune(nn.Module):

    def __init__(self, n_classes=1):
        super(AlexNet_finetune, self).__init__()
        self.features = models.alexnet(pretrained=True).features
        self.linear = torch.nn.Linear(9216, n_classes)

    def forward(self, x,):
        feature_x = self.features(x).view(x.shape[0], -1)
        return torch.sigmoid(self.linear(feature_x))

