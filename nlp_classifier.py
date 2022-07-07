import torch
import torch.nn as nn
from arcface import ArcMarginProduct
from transformer_emb import TransformerEmb

class NlpClassifier(nn.Module):
    def __init__(self, pretrained_model, num_labels, emb_size=128, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.num_labels = num_labels
        self.emb_size = emb_size
        self.emb_layer = TransformerEmb(self.ptm, self.emb_size)
        self.classifier = ArcMarginProduct(self.ptm.config.hidden_size, self.num_labels)

    def forward(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                label=None,
                is_test=False):
        query_token_embedding = self.emb_layer(query_input_ids,
                                               query_token_type_ids,
                                               query_position_ids,
                                               query_attention_mask)
        if not is_test:
            return self.classifier(query_token_embedding, label)
        else:
            return self.classifier.forward_test(query_token_embedding)

    def predict_emb(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):

        return self.emb_layer(query_input_ids,
                              query_token_type_ids,
                              query_position_ids,
                              query_attention_mask)