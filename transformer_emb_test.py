from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from functools import partial

_P = os.path.dirname
dsf_root = _P(os.path.realpath(__file__))
sys.path.append(dsf_root)

import torch
import unittest
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
from transformer_emb import TransformerEmb
from torch.utils.data import DataLoader
from datasets import load_dataset


class TestTransformerEmb(unittest.TestCase):
    def setUp(self):
        self.pretrained_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        self.model = TransformerEmb(self.pretrained_model)

    def tokenize_function(self,example):
        return self.tokenizer(text=example["spuname"], padding="max_length", max_length=128, truncation=True)

    def test_forward(self):
        # test print token
        query_encoded_inputs = self.tokenizer(text="长沙是臭豆腐之都[SEP]哈尔滨是腊肠之都", padding="max_length", max_length=512, truncation=True)
        query_input_ids = query_encoded_inputs["input_ids"]
        query_token_type_ids = query_encoded_inputs["token_type_ids"]
        print(query_input_ids)
        print(query_token_type_ids)

        # test custom data
        data_files = {"train": "test.csv", "test": "test.csv"}
        raw_datasets = load_dataset("csv", data_files=data_files, delimiter=",")
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        #post process
        tokenized_datasets = tokenized_datasets.remove_columns(["spuname"])
        tokenized_datasets = tokenized_datasets.rename_column("category_name_id", "labels")
        tokenized_datasets.set_format("torch")

        print(tokenized_datasets["train"].column_names)

        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
        )

        for batch in train_dataloader:
            print({k: v.shape for k, v in batch.items()})
            break

        for step, batch in enumerate(train_dataloader, start=1):
            #labels, query_input_ids, query_token_type_ids, query_attention_mask = batch
            emb = self.model(query_input_ids=batch['input_ids'],
                             query_token_type_ids=batch['token_type_ids'],
                             query_attention_mask=batch['attention_mask'])
            print(emb.shape)
            break