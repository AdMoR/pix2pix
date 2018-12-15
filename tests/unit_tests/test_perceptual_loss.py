from unittest import TestCase

import torch
import torchvision

from nn_utils.model import EncoderNet
from nn_utils.perceptual_loss import PerceptualLoss


class TestPerceptualLoss(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.loss = PerceptualLoss(model=torchvision.models.alexnet)

    def test_basic_forward(self):
        loss = self.loss.forward(torch.randn((1, 3, 224, 224)), torch.randn((1, 3, 224, 224)))
        self.assertGreaterEqual(float(loss), 0)

    def test_list_forward(self):
        tensor_list = [torch.randn((1, 3, 224, 224)), torch.randn((1, 3, 224, 224))]
        loss = self.loss.forward(tensor_list, tensor_list)
        self.assertEquals(float(loss), 0)
        

