from unittest import TestCase

import torch

from nn_utils.model import UNet, EncoderNet
from nn_utils.gan_losses import AdversarialConditionalLoss


class TestAdvLoss(TestCase):

    def setUp(self):
        self.gen = UNet([3, 10, 20, 40])
        self.disc = EncoderNet([6, 20, 40, 60])
        self.adv_loss = AdversarialConditionalLoss(self.gen, self.disc, loss="L2")

    def test_full_forward(self):
        x = torch.randn((1, 3, 224, 224))
        y = torch.randn((1, 3, 224, 224))

        gen_loss = self.adv_loss.generator_forward(x, y, None)
        print("gen loss", gen_loss)

    def test_backward(self):
        x = torch.randn((1, 3, 224, 224))
        y = torch.randn((1, 3, 224, 224))
        gen_optimizer = torch.optim.SGD(self.gen.parameters(), lr=0.0001)
        disc_optimizer = torch.optim.SGD(self.disc.parameters(), lr=0.0001)

        disc_loss = self.adv_loss.discriminator_forward(x, y, None)
        gen_loss = self.adv_loss.generator_forward(x, y, None)

        gen_loss.backward()
        gen_optimizer.step()

        disc_loss.backward()
        disc_optimizer.step()


