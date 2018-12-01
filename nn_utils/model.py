import torch


class BasicEncoderBlock(torch.nn.Module):

    def __init__(self, in_, out_, act_fn, pool_fn=torch.nn.MaxPool2d):
        super(BasicEncoderBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_, out_, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_)
        self.act_fn = act_fn
        if pool_fn is not None:
            self.pool = pool_fn(kernel_size=(2, 2), stride=2)
        else:
            self.pool = None

    def forward(self, x):
        if self.pool is not None:
            return self.pool(self.act_fn(self.bn(self.conv(x))))
        else:
            return self.act_fn(self.bn(self.conv(x)))


class BasicDecoderBlock(torch.nn.Module):

    def __init__(self, in_, out_, act_fn):
        super(BasicDecoderBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_, out_, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_)
        self.act_fn = act_fn

    def forward(self, x, previous):
        x_prime = torch.nn.functional.interpolate(x, scale_factor=(2, 2))
        combined = torch.cat([x_prime, previous], dim=1)
        return self.act_fn(self.bn(self.conv(combined)))


class UNet(torch.nn.Module):

    def __init__(self, layers):
        super(UNet, self).__init__()
        self.act_fn = torch.nn.functional.relu
        self.encoder = dict()
        self.decoder = dict()

        for i, (in_, out_) in enumerate(zip(layers[:-1], layers[1:])):
            self.encoder[i] = BasicEncoderBlock(in_, out_, self.act_fn)
        for i, (in_, out_) in enumerate(zip(layers[1:], layers[:-1])):
            self.decoder[i] = BasicDecoderBlock(in_ + out_, out_, self.act_fn)

    def forward(self, x):
        inputs = dict()
        for i in range(len(self.encoder)):
            print(x.shape)
            inputs[i] = x
            x = self.encoder[i](x)

        print(self.encoder, self.decoder, {i: inputs[i].shape for i in inputs})
        for i in range(len(self.encoder)):
            print(x.shape, i)
            x = self.decoder[len(self.encoder) - (i + 1)](x, inputs[len(self.encoder) - (i + 1)])
        return x

