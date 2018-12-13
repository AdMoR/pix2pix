from unittest import TestCase

import torch

from nn_utils.model import UNet, EncoderNet


class TestUNet(TestCase):

    def setUp(self):
        self.model = UNet([3, 10, 20, 40])

    def test_forward(self):
        res = self.model.forward(torch.randn((1, 3, 2**10, 2**10)))
        self.assertEqual(res.shape, (1, 3, 2**10, 2**10), "Bad shape")


class TestEncoder(TestCase):

    def setUp(self):
        self.model = EncoderNet([6, 10, 20, 40])

    def test_encoder_net(self):
        res = self.model.forward(torch.randn((2, 3, 224, 224)), torch.randn((2, 3, 224, 224)))
        self.assertEqual(res.shape, (2, 1, 26, 26), "Bad shape")
