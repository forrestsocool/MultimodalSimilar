from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from functools import partial

_P = os.path.dirname
dsf_root = _P(os.path.realpath(__file__))
sys.path.append(dsf_root)

import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from image_emb import ImageEmb

class TestImageEmb(unittest.TestCase):
    def setUp(self):
        self.pretrained_model = timm.create_model('efficientnet_b4', pretrained=True)
        self.model = ImageEmb(self.pretrained_model)
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)

    # def tokenize_function(self,example):
    #     return self.tokenizer(text=example["spuname"], padding="max_length", max_length=128, truncation=True)

    def test_forward(self):
        # test print token
        img = Image.open("./11.jpg").convert('RGB')
        rgb_tensor = self.transform(img)
        print(rgb_tensor.shape)
        rgb_tensor = rgb_tensor.unsqueeze(0)
        print(rgb_tensor.shape)
        print(self.model(rgb_tensor))