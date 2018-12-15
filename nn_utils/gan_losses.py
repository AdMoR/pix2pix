import torch


label_type = torch.int64


class AdversarialConditionalLoss(torch.nn.Module):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def __init__(self, generator, discriminator, device=torch.device('cpu'), loss="L2"):
        super(AdversarialConditionalLoss, self).__init__()
        self.device = device
        self.gen = generator
        self.dis = discriminator
        self.lambda_ = 10
        if loss == "L2":
            self.loss = lambda x, real:\
                torch.norm(x - int(real) * torch.ones_like(x).to(self.device), 2)
        else:
            self.loss = lambda x, real:\
                -torch.mean(torch.log(x)) if real\
                else -torch.mean(torch.ones_like(x).to(self.device) - torch.log(x))

    def fake_or_real_forward(self, x, y, real=True):
        discriminator_value, dis_layers = self.dis(y, x, keep_intermediate=True)
        return self.loss(discriminator_value, real), dis_layers

    def regularization(self, y, y_hat):
        return (1. / y.shape[0]) * torch.norm(y - y_hat, 1)

    def discr_layer_regularization(self, fake_features, real_features):
        return (1. / len(fake_features)) * sum(
            [torch.norm(fake_features[i] - real_features[i], 1)
             for i in range(len(fake_features))],
            torch.zeros(1).to(self.device)
        )

    def discriminator_forward(self, x, y, z, verbose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        fake_sample_loss, dis_fake_layers = self.fake_or_real_forward(x, y_hat, real=False)
        real_sample_loss, dis_real_layers = self.fake_or_real_forward(x, y, real=True)

        if verbose:
            print("fake : ", fake_sample_loss, "real ", real_sample_loss)
        discriminator_loss = fake_sample_loss + real_sample_loss

        # According to the paper, discriminator has its loss divided by 2
        return torch.mean(discriminator_loss)

    def generator_forward(self, x, y, z, verbose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        fake_sample_loss, dis_fake_layers = self.fake_or_real_forward(x, y_hat, real=True)
        _, dis_real_layers = self.fake_or_real_forward(x, y, real=True)

        # We can do a simple L1 loss btw original image and generated
        if not dis_real_layers:
            regularisation = self.lambda_ * self.regularization(y, y_hat)
        # Or compare all the intermediate layers of the discriminator
        else:
            regularisation = self.lambda_ *\
                self.discr_layer_regularization(dis_fake_layers, dis_real_layers)

        if verbose:
            print("fake : ", fake_sample_loss, "reg ", regularisation)
        generator_loss = -fake_sample_loss + regularisation

        return torch.mean(generator_loss)


class AdversarialLoss(AdversarialConditionalLoss):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def fake_or_real_forward(self, x, y, real=True):
        discriminator_value = self.dis(y)
        if real:
            return self.loss(discriminator_value)
        else:
            return self.loss(torch.ones_like(discriminator_value) - discriminator_value)

