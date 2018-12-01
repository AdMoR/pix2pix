from unittest import TestCase

import torch

from nn_utils.model import UNet


class TestUNet(TestCase):

    def setUp(self):
        self.model = UNet([3, 10, 20, 40])

    def test_forward(self):
        res = self.model.forward(torch.randn((1, 3, 224, 224)))
        self.assertEqual(res.shape, (1, 3, 224, 224), "Bad shape")