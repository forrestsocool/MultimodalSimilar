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
    def __init__(self, model_name, fc_dim, num_labels,
                 m=0.2,
                 pretrained=True,
                 use_fc=True):
        super(CvClassifier, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        # we will put FC layers over backbone to classfy images based on label groups
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.use_fc = use_fc
        self.num_labels = num_labels

        # build top fc layers
        if self.use_fc:
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(in_features,fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            in_features = fc_dim
        self.classifier = ArcMarginProduct(in_features, self.num_labels, m=m)

    def forward(self, input:torch.Tensor, label=None, is_test=False):
        img_embedding = self.predict_emb(input)
        if not is_test:
            return self.classifier(img_embedding, label)
        else:
            return self.classifier.forward_test(img_embedding)

    def predict_emb(self, inp:torch.Tensor):
        batch_dim = inp.shape[0]
        inp = self.backbone(inp)
        inp = self.pooling(inp).view(batch_dim, -1)
        if self.use_fc:
            inp = self.dropout(inp)
            inp = self.fc(inp)
            inp = self.bn(inp)
        return inp
