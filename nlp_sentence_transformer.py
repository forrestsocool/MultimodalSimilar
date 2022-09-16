import torch
import torch.nn as nn
from arcface import ArcMarginProduct
from transformer_emb import TransformerEmb

class NlpSentenceTransformer(nn.Module):
    def __init__(self, pretrained_model, emb_size=128, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        # self.num_labels = num_labels
        self.emb_size = emb_size
        self.emb_layer = TransformerEmb(self.ptm, self.emb_size)
        #self.classifier = ArcMarginProduct(self.ptm.config.hidden_size, self.num_labels)
        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config.hidden_size * 3, 2)

    def forward(self,
                query_input_ids,
                title_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None,
                title_token_type_ids=None,
                title_position_ids=None,
                title_attention_mask=None,
                label=None):
        query_token_embedding = self.emb_layer(query_input_ids,
                                               query_token_type_ids,
                                               query_position_ids,
                                               query_attention_mask)
        title_token_embedding = self.emb_layer(title_input_ids,
                                               title_token_type_ids,
                                               title_position_ids,
                                               title_attention_mask)

        sub = torch.abs(torch.subtract(query_token_embedding, title_token_embedding))
        projection = torch.cat((query_token_embedding, title_token_embedding, sub), dim=-1)
        logits = self.classifier(projection)
        return logits

    def predict_emb(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):

        return self.emb_layer(query_input_ids,
                              query_token_type_ids,
                              query_position_ids,
                              query_attention_mask)