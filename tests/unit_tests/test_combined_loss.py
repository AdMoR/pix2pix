from unittest import TestCase

import torch, torchvision

from nn_utils.model import UNet, EncoderNet
from nn_utils.gan_losses import AdversarialConditionalLoss
from nn_utils.perceptual_loss import PerceptualLoss
from nn_utils.combined_loss import CombinedGANLoss


class TestCombinedGanLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.gen = UNet([3, 10, 20, 40])
        cls.disc_1 = EncoderNet([6, 20, 40, 60])
        cls.disc_2 = EncoderNet([6, 20, 40, 60])
        cls.adv_loss_1 = AdversarialConditionalLoss(cls.gen, cls.disc_1, loss="L2")
        cls.adv_loss_2 = AdversarialConditionalLoss(cls.gen, cls.disc_2, loss="L2")
        cls.perceptual_loss = PerceptualLoss(model=torchvision.models.alexnet)
        cls.loss = CombinedGANLoss(gan_losses=[cls.adv_loss_1, cls.adv_loss_2])

    def test_basic_forward(self):
        self.loss.generator_forward(torch.randn((1, 3, 224, 224)), torch.randn((1, 3, 224, 224)))
