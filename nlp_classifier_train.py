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
from datasets import load_dataset,load_metric
from nlp_classifier import NlpClassifier
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

pretrained_model=None
tokenizer = None
model = None
batch_size = 256
num_epochs = 30

stopwords=pd.read_csv("./stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']

#分词去停用词，并整理为fasttext要求的文本格式
def preprocess_for_infer(spu_names):
    result=[]
    for spu_name in spu_names:
        line = spu_name
        for r in remove_words:
            line = line.replace(r, '')
        commas = re.findall('\[[^()]*\]', line)
        for c in commas:
            line = line.replace(c, '')
        result.append(line)
    return result

def tokenize_function(example):
    return tokenizer(text=preprocess_for_infer(example["spu_name"]), padding="max_length", max_length=128, truncation=True)

# def compute_metrics(logits, labels):
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    # load pretrained model
    pretrained_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = NlpClassifier(pretrained_model, 796)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # read custom data
    data_files = {
        "train": "/home/ma-user/work/MultimodalSimilar/yuanben_spusn_20220623/train_nlp_only.csv",
        "test": "/home/ma-user/work/MultimodalSimilar/yuanben_spusn_20220623/test_nlp_only.csv"}
    raw_datasets = load_dataset("csv", data_files=data_files, delimiter=",")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # post process
    tokenized_datasets = tokenized_datasets.remove_columns(["spu_name"])
    tokenized_datasets = tokenized_datasets.rename_column("cateid", "labels")
    tokenized_datasets.set_format("torch")

    #train params
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    num_training_steps = num_epochs * len(train_dataloader)
    optimizer_emb = AdamW(model.emb_layer.parameters(), lr=5e-5)
    lr_scheduler_emb = get_scheduler(
        name="linear", optimizer=optimizer_emb, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    optimizer_fc = AdamW(model.classifier.parameters(), lr=1e-2)
    lr_scheduler_fc = get_scheduler(
        name="linear", optimizer=optimizer_fc, num_warmup_steps=0.15*num_training_steps, num_training_steps=num_training_steps
    )

    #train loop
    entroy=nn.CrossEntropyLoss()
    progress_bar = tqdm(range(num_training_steps))

    test_acc = 0
    global_step = 0
    train_accuracy = torchmetrics.Accuracy()
    test_accuracy = torchmetrics.Accuracy()
    train_accuracy.to(device)
    test_accuracy.to(device)

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader, start=1):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            # labels, query_input_ids, query_token_type_ids, query_attention_mask = batch
            labels=batch['labels']
            preds = model(query_input_ids=batch['input_ids'],
                        query_token_type_ids=batch['token_type_ids'],
                        query_attention_mask=batch['attention_mask'],
                        label=batch['labels'])
            loss = entroy(preds, labels)
            loss.backward()

            predictions = torch.argmax(preds, dim=-1)
            train_accuracy(predictions, labels)


            optimizer_emb.step()
            lr_scheduler_emb.step()
            optimizer_emb.zero_grad()

            optimizer_fc.step()
            lr_scheduler_fc.step()
            optimizer_fc.zero_grad()

            train_acc = train_accuracy.compute().cpu().detach().numpy()
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Acc/train', train_acc, global_step)

            progress_bar.set_postfix(loss=f'%.2f' % loss.cpu().detach().numpy(), acc=f'%.2f'%train_acc, test_acc=f'%.2f'%test_acc)
            progress_bar.update(1)
            global_step += 1
            if global_step % 100 == 0:
                model.eval()
                for batch in test_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        preds = model(query_input_ids=batch['input_ids'],
                                      query_token_type_ids=batch['token_type_ids'],
                                      query_attention_mask=batch['attention_mask'],
                                      label=batch['labels'],
                                      is_test=True)
                    test_labels = batch['labels']
                    predictions = torch.argmax(preds, dim=-1)
                    test_accuracy(predictions, test_labels)
                    test_acc = test_accuracy.compute().cpu().detach().numpy()
                writer.add_scalar('Acc/test', test_acc, global_step)
            # save model
            if global_step % 1000 == 0:
                torch.save(model,'./nlp_model/{}.pt'.format(global_step))
    # model.eval()
    # for batch in test_dataloader:
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         preds = model(query_input_ids=batch['input_ids'],
    #                       query_token_type_ids=batch['token_type_ids'],
    #                       query_attention_mask=batch['attention_mask'],
    #                       label=batch['labels'],
    #                       is_test=True)
    #
    #     predictions = torch.argmax(preds, dim=-1)
    #     metric_test.add_batch(predictions=predictions, references=batch["labels"])
    # metric_test_result = metric_test.compute()
    # print(metric_test_result)
