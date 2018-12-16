import torch

from .gan_losses import AdversarialConditionalLoss
from .perceptual_loss import PerceptualLoss


class CombinedGANLoss(torch.nn.Module):

    def __init__(self, gan_losses=[], device=None):
        self.device = device
        self.gan_losses = gan_losses
        self.lambda_reg = 10

    def disc_parameters(self):
        return sum([list(loss.dis.parameters()) for loss in self.gan_losses], [])

    def discriminator_forward(self, x, y, z=None):
        return sum([loss(x, y, z, discriminator=True)[0] for loss in self.gan_losses], torch.zeros(1).to(self.device))

    def generator_forward(self, x, y, z=None):
        total_loss = None
        for loss in self.gan_losses:
            gen_gan_loss, fake_layers, real_layers = loss.forward(x, y, z, discriminator=False)
            for fake_layer, real_layer in zip(fake_layers.values(), real_layers.values()):
                gen_gan_loss += self.lambda_reg * torch.norm(fake_layer - real_layer, 1)

            if total_loss is None:
                total_loss = gen_gan_loss
            else:
                total_loss += gen_gan_loss

        return total_loss
