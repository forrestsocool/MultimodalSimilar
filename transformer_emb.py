from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEmb(nn.Module):
    def __init__(self, pretrained_model, emb_size=128, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(p=dropout if dropout is not None else 0.1)
        # num_labels = 2 (similar or dissimilar)
        self.emb_layer = nn.Linear(self.ptm.config.hidden_size, emb_size)
        self.bn_layer = nn.BatchNorm1d(self.ptm.config.hidden_size)

    def forward(self,
                query_input_ids,
                query_token_type_ids=None,
                query_position_ids=None,
                query_attention_mask=None):
        outputs = self.ptm(input_ids=query_input_ids,
                                 attention_mask=query_attention_mask,
                                 token_type_ids=query_token_type_ids,
                                 position_ids=query_position_ids)
        pooled_out = outputs.pooler_output
        query_token_embedding = self.dropout(pooled_out)
        #return self.emb_layer(query_token_embedding)
        batch_normed_embedding = self.bn_layer(query_token_embedding)
        normed_embedding = F.normalize(batch_normed_embedding,p=2,dim=1)
        return normed_embedding

    # def forward_test(self,
    #             query_input_ids,
    #             query_token_type_ids=None,
    #             query_position_ids=None,
    #             query_attention_mask=None):
    #     outputs = self.ptm(input_ids=query_input_ids,
    #                        attention_mask=query_attention_mask,
    #                        token_type_ids=query_token_type_ids,
    #                        position_ids=query_position_ids)
    #     pooled_out = outputs.pooler_output
    #     query_token_embedding = self.dropout(pooled_out)
    #     # return self.emb_layer(query_token_embedding)
    #     batch_normed_embedding = self.bn_layer(query_token_embedding)
    #     normed_embedding = F.normalize(batch_normed_embedding, p=2, dim=1)
    #     return normed_embedding