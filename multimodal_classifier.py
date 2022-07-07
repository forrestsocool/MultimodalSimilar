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

class MultimodalClassifier(nn.Module):
    def __init__(self, device, cv_classifier_path, nlp_classifier_path, emb_size, num_labels, dropout=None):
        super().__init__()
        self.cv = torch.load(cv_classifier_path)
        self.nlp = torch.load(nlp_classifier_path)
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.num_labels = num_labels
        self.emb_size = emb_size
        self.classifier = ArcMarginProduct(in_feature=self.emb_size, out_feature=self.num_labels, m=0.5)
        self.cv.to(device)
        self.nlp.to(device)
        self.classifier.to(device)

    def forward(self,
                img_input:torch.Tensor,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                label=None,
                is_test=False):
        final_embdding = self.predict_emb(img_input=img_input,
                                          query_input_ids=query_input_ids,
                                          query_token_type_ids=query_token_type_ids,
                                          query_attention_mask=query_attention_mask)
        if not is_test:
            return self.classifier(final_embdding, label)
        else:
            return self.classifier.forward_test(final_embdding)

    def predict_emb(self,
                    img_input: torch.Tensor,
                    query_input_ids,
                    query_token_type_ids=None,
                    query_position_ids=None,
                    query_attention_mask=None):
        img_embedding = self.cv.predict_emb(img_input)
        title_embedding = self.nlp.predict_emb(query_input_ids=query_input_ids,
                        query_token_type_ids=query_token_type_ids,
                        query_attention_mask=query_attention_mask)
        img_embedding_norm = F.normalize(img_embedding,p=2,dim=1)
        title_embedding_norm = F.normalize(title_embedding,p=2,dim=1)
        final_embdding = torch.cat((img_embedding_norm, title_embedding_norm), 1)
        return final_embdding