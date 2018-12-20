from unittest import TestCase

import torch

from dataset_handler.ade20k_loader import EdgeADE20k


class TestADE20k(TestCase):

    def setUp(self):
        self.data = EdgeADE20k("/Users/amorvan/mount_ubuntu/databases/ADE20K_2016_07_26/", is_transform=True)

    def test(self):
        x, y = self.data.__getitem__(0)
        y = y.long()
        # Verify that all elements of y are 0 or 1
        self.assertTrue(torch.min(((y == 1) + (y == 0)) == torch.ones_like(y).byte()), 1)
