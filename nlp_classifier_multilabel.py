import torch
import torch.nn as nn
from arcface import ArcMarginProduct
from transformer_emb import TransformerEmb

class NlpClassifierMultilabel(nn.Module):
    def __init__(self, pretrained_model, firstcate_num_labels, secondcate_num_labels, tag_num_labels, emb_size=128, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.emb_size = emb_size
        # self.emb_layer = TransformerEmb(self.ptm, self.emb_size)
        self.emb_layer = TransformerEmb(self.ptm)
        self.firstcate_classifier = ArcMarginProduct(self.ptm.config.hidden_size, firstcate_num_labels, m=0.4)
        self.secondcate_classifier = ArcMarginProduct(self.ptm.config.hidden_size, secondcate_num_labels, m=0.2)
        self.tag_classifier = ArcMarginProduct(self.ptm.config.hidden_size, tag_num_labels, m=0.1)

    def forward(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                firstcate_label=None,
                secondcate_label=None,
                tag_label=None,
                is_test=False):
        query_token_embedding = self.emb_layer(query_input_ids,
                                               query_token_type_ids,
                                               query_position_ids,
                                               query_attention_mask)
        if not is_test:
            return (self.firstcate_classifier(query_token_embedding, firstcate_label),
                    self.secondcate_classifier(query_token_embedding, secondcate_label),
                    self.tag_classifier(query_token_embedding, tag_label))
        else:
            return (self.firstcate_classifier.forward_test(query_token_embedding),
                    self.secondcate_classifier.forward_test(query_token_embedding),
                    self.tag_classifier.forward_test(query_token_embedding))

    def predict_emb(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):

        return self.emb_layer(query_input_ids,
                              query_token_type_ids,
                              query_position_ids,
                              query_attention_mask)