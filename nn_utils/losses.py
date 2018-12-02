import torch


label_type = torch.int64


class AdversarialConditionalLoss(torch.nn.Module):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def __init__(self, generator, discriminator, device=torch.device('cpu')):
        super(AdversarialConditionalLoss, self).__init__()
        self.device = device
        self.gen = generator
        self.dis = discriminator
        self.lambda_ = 0.1
        self.loss = torch.nn.CrossEntropyLoss()

    def fake_or_real_forward(self, x, y, labels):
        discriminator_labels = self.dis(y, x)
        loss_fake = self.loss(discriminator_labels, labels)
        return loss_fake

    def regularization(self, y, y_hat):
        return (1. / y.shape[0]) * torch.norm(y - y_hat, 1)

    def discriminator_forward(self, x, y, z, verbose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        fake_sample_loss = self.fake_or_real_forward(x, y_hat, torch.zeros(x.shape[0], dtype=label_type).to(device=self.device))
        real_sample_loss = self.fake_or_real_forward(x, y, torch.ones(x.shape[0], dtype=label_type).to(device=self.device))

        if verbose:
            print("fake : ", fake_sample_loss, "real ", real_sample_loss)
        discriminator_loss = fake_sample_loss + real_sample_loss

	# According to the paper, discriminator has its loss divided by 2
        return discriminator_loss

    def generator_forward(self, x, y, z, verbose=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        fake_sample_loss = self.fake_or_real_forward(x, y_hat, torch.zeros(x.shape[0], dtype=label_type).to(device=self.device))
        regularisation = self.lambda_ * self.regularization(y, y_hat)

        if verbose:
            print("fake : ", fake_sample_loss, "reg ", regularisation)
        generator_loss = -fake_sample_loss + regularisation

	# According to the paper, discriminator has its loss divided by 2
        return generator_loss


class AdversarialLoss(AdversarialConditionalLoss):
    """
    Works for generator going from domain X to domain Y
    Discriminator checks that the x, y belongs to the a real sample
    """

    def fake_or_real_forward(self, x, y, labels):
        discriminator_labels = self.dis(y)
        loss_fake = self.loss(discriminator_labels, labels)
        return loss_fake

