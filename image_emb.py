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
from arcface import ArcMarginProduct


class ImageEmb(nn.Module):
    def __init__(self, pretrained_model, use_bn=False):
        super().__init__()
        self.ptm = pretrained_model
        self.bn_layer = nn.BatchNorm1d(self.ptm.classifier.in_features)
        #只输出embedding
        self.ptm.reset_classifier(0)
        self.use_bn = use_bn

        # #固定预训练权重
        # for p in self.ptm.parameters():
        #     p.requires_grad = False

    def forward(self, rgb_tensor):
        outputs = self.ptm(rgb_tensor)
        # if self.use_bn:
        #     outputs = self.bn_layer(outputs)
        # normed_op = F.normalize(outputs, p=2, dim=1)  # 对指定维度进行运算
        # return normed_op

        return outputs