import torch


label_type = torch.int64


class AdversarialConditionalLoss(torch.nn.Module):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def __init__(self, generator, discriminator, device=torch.device('cpu'), loss=None):
        super(AdversarialConditionalLoss, self).__init__()
        self.device = device
        self.gen = generator
        self.dis = discriminator
        self.lambda_ = 100
        if loss == "L2":
            self.loss = lambda x: torch.norm(x, 2)
        else:
            self.loss = lambda x: -torch.mean(torch.log(x))

    def fake_or_real_forward(self, x, y, real=True):
        discriminator_value = self.dis(y, x)
        if real:
            return -self.loss(discriminator_value)
        else:
            return -self.loss(1 - discriminator_value)

    def regularization(self, y, y_hat):
        return (1. / y.shape[0]) * torch.norm(y - y_hat, 1)

    def discriminator_forward(self, x, y, z, verbose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        fake_sample_loss = self.fake_or_real_forward(x, y_hat, real=False)
        real_sample_loss = self.fake_or_real_forward(x, y, real=True)

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

        fake_sample_loss = self.fake_or_real_forward(x, y_hat, real=True)
        regularisation = self.lambda_ * self.regularization(y, y_hat)

        if verbose:
            print("fake : ", fake_sample_loss, "reg ", regularisation)
        generator_loss = -fake_sample_loss + regularisation

	# According to the paper, discriminator has its loss divided by 2
        return torch.mean(generator_loss)


class AdversarialLoss(AdversarialConditionalLoss):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def fake_or_real_forward(self, x, y, real=True):
        discriminator_value = self.dis(y)
        if real:
            return -torch.mean(torch.log(discriminator_value))
        else:
            return -torch.mean(torch.log(torch.ones_like(discriminator_value) - discriminator_value))

