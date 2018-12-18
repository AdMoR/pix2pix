from unittest import TestCase

from dataset_handler.ade20k_loader import EdgeADE20k


class TestADE20k(TestCase):

    def setUp(self):
        self.data = EdgeADE20k("/data/ADE20K_2016_07_26/")

    def test(self):
        x, y = self.data.__getitem__(0)
        print(x.shape, y.shape)
