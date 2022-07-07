import torch
import torch.nn as nn
from arcface import ArcMarginProduct
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
from image_emb import ImageEmb

class CvClassifier(nn.Module):
    def __init__(self, pretrained_model, emb_size, num_labels, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.num_labels = num_labels
        self.emb_size = emb_size
        self.emb_layer = ImageEmb(self.ptm, use_bn=True)
        self.classifier = ArcMarginProduct(self.emb_size, self.num_labels)

    def forward(self, input:torch.Tensor, label=None, is_test=False):
        img_embedding = self.emb_layer(input)
        if not is_test:
            return self.classifier(img_embedding, label)
        else:
            return self.classifier.forward_test(img_embedding)

    def predict_emb(self, transformed_tensor:torch.Tensor):
        return self.emb_layer(transformed_tensor)
