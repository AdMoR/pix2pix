import torch
import torch.nn as nn
from torchvision import models


class BasicEncoderBlock(nn.Module):

    def __init__(self, in_, out_, act_fn=nn.ReLU, pool_fn=nn.MaxPool2d, stride=1):
        super(BasicEncoderBlock, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_, out_, 3, padding=1, stride=stride),
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
            nn.Conv2d(in_, out_, 3, padding=1),
            nn.BatchNorm2d(out_),
            act_fn(inplace=True)
        )

    def forward(self, x, previous):
        x_prime = nn.functional.interpolate(x, scale_factor=(2, 2))
        combined = torch.cat([x_prime, previous], dim=1)
        return self.op(combined)


class UNet(torch.nn.Module):

    def __init__(self, layers, target_dim=3):
        super(UNet, self).__init__()
        self.act_fn = nn.ReLU
        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            self.encoder[str(i)] = BasicEncoderBlock(in_, out_, self.act_fn)
        for i, (in_, out_) in enumerate(zip(layers[1:], layers[:-1])):
            in_ += out_
            if i == 0:
                out_ = target_dim
            self.decoder[str(i)] = BasicDecoderBlock(in_, out_, self.act_fn)

    def forward(self, x, z=None):
        inputs = dict()
        for i in range(len(self.encoder)):
            inputs[i] = x
            x = self.encoder[str(i)](x)

        for i in range(len(self.encoder)):
            x = self.decoder[str(len(self.encoder) - (i + 1))](x, inputs[len(self.encoder) - (i + 1)])
        return torch.tanh(x)


class EncoderNet(nn.Module):

    def __init__(self, layers, n_classes=1):
        super(EncoderNet, self).__init__()
        self.act_fn = nn.LeakyReLU
        self.encoder = nn.ModuleDict()

        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            self.encoder[str(i)] = BasicEncoderBlock(in_, out_, self.act_fn)
        self.linearizer = nn.Conv2d(layers[-1], n_classes, 3)

    def forward(self, x, y=None, keep_intermediate=False):
        outputs = {}
        if y is not None:
            z = torch.cat([x, y], dim=1)
        else:
            z = x
        for i in range(len(self.encoder)):
            if keep_intermediate:
                outputs[i] = z
            z = self.encoder[str(i)](z)

        z = self.linearizer(z)

        return torch.sigmoid(z), outputs


class ResidualTransformer(nn.Module):

    def __init__(self, encoder_layers=[], residual_layers=[]):
        decoder_layers = list(reversed(encoder_layers))
        self.encoder = torch.Sequential(
            [
                BasicEncoderBlock(
                    encoder_layers[i],
                    encoder_layers[i + 1],
                    pool_fn=None,
                    stride=1 if i == 0 else 2
                )
                for i in range(len(encoder_layers) - 1)
            ]
        )
        self.res_block = torch.Sequential(
            [
                BasicEncoderBlock(
                    residual_layers[i],
                    residual_layers[i + 1],
                    pool_fn=None,
                    stride=1
                )
                for i in range(len(residual_layers) - 1)
            ]
        )
        self.decoder = torch.Sequential(
            [
                BasicEncoderBlock(
                    residual_layers[i + 1],
                    residual_layers[i],
                    pool_fn=None,
                    stride=1 if i == 0 else 0.5
                )
                for i in reversed(range(len(residual_layers) - 1))
            ]
        )
        self.process = torch.Sequential([self.encoder, self.res_block, self.decoder])

    def forward(self, x):
        return self.process(x)

