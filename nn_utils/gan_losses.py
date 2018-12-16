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
                -torch.mean(torch.log(torch.sigmoid(x))) if real\
                else -torch.mean(torch.ones_like(x).to(self.device) - torch.log(torch.sigmoid(x)))

    def fake_or_real_forward(self, x, y, real=True):
        discriminator_value, dis_layers = self.dis(y, x, keep_intermediate=True)
        return self.loss(discriminator_value, real), dis_layers

    def forward(self, x, y, z, discriminator, verbose=False):
        """
        returns: gan loss for discr or gen, intermediate layer for fake and real samples of the discriminator
        """
        x = x.to(self.device)
        y = y.to(self.device)
        if z is not None:
            z = z.to(self.device)

        y_hat = self.gen.forward(x, z)

        # We get the gan loss 
        fake_sample_loss, dis_fake_layers = self.fake_or_real_forward(x, y_hat, real=not discriminator)
        real_sample_loss, dis_real_layers = self.fake_or_real_forward(x, y, real=True)

        if verbose:
            if discriminator:
                print("fake : ", fake_sample_loss, "real ", real_sample_loss)
            else:
                print("fake : ", fake_sample_loss, "reg ", regularisation)

        generator_loss = -fake_sample_loss
        discriminator_loss = fake_sample_loss + real_sample_loss

        if discriminator:
            return discriminator_loss, dis_fake_layers, dis_real_layers
        else:
            return generator_loss, dis_fake_layers, dis_real_layers


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

