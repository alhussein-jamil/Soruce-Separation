import unittest

import torch
from torch import nn

from vocal_sep.models.unet import MaskUNet


class TestMaskUNet(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MaskUNet().to(self.device)

    def test_forward_shape(self):
        sample = torch.rand(2, 1, 512, 128, device=self.device)
        output = self.model(sample)
        self.assertEqual(output.shape, sample.shape)

    def test_output_is_mask(self):
        sample = torch.rand(1, 1, 512, 128, device=self.device)
        output = self.model(sample)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
        self.assertIsInstance(self.model, nn.Module)


if __name__ == "__main__":
    unittest.main()
