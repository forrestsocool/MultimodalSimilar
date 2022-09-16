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
from nlp_classifier_multilabel import NlpClassifierMultilabel
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import argparse

pretrained_model=None
tokenizer = None
model = None
batch_size = 2*8*128
num_epochs = 30

# stopwords=pd.read_csv("/home/ma-user/modelarts/inputs/data_url_1/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# stopwords=stopwords['stopword'].values
# remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']

# #分词去停用词，并整理为fasttext要求的文本格式
# def preprocess_for_infer(spu_names):
#     result=[]
#     for spu_name in spu_names:
#         line = spu_name
#         for r in remove_words:
#             line = line.replace(r, '')
#         commas = re.findall('\[[^()]*\]', line)
#         for c in commas:
#             line = line.replace(c, '')
#         result.append(line)
#     return result

def tokenize_function(example):
    return tokenizer(text=example["title"], padding="max_length", max_length=128, truncation=True)

# def compute_metrics(logits, labels):
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
def get_class_weights(data):
    weight_dict=dict()
    # Format of row : PostingId, Image, ImageHash, Title, LabelGroup
    # LabelGroup index is 4 and it is representating class information
    for i, row in data.iterrows():
        weight_dict[row['lv2_category_id']]=0
    # Word dictionary keys will be label and value will be frequency of label in dataset
    for i, row in data.iterrows():
        weight_dict[row['lv2_category_id']]+=1
    # for each data point get label count data
    class_sample_count= np.array([weight_dict[row['lv2_category_id']] for i, row in data.iterrows()])
    # each data point weight will be inverse of frequency
    weight = 1. / class_sample_count
    weight=torch.from_numpy(weight)
    return weight

lv1_weight = 10
lv2_weight = 5
tag_weight = 1

parser = argparse.ArgumentParser()
parser.add_argument("--lv1_weight", type=float, default=10.0)
parser.add_argument("--lv2_weight", type=float, default=5.0)
parser.add_argument("--tag_weight", type=float, default=1.0)
parser.add_argument("--data_input", type=str)
parser.add_argument("--data_url", type=str)
parser.add_argument("--huggingface", type=str)
parser.add_argument("--data_output", type=str)
args = parser.parse_args()


if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    # load pretrained model
    pretrained_model = BertModel.from_pretrained(args.huggingface+'roberta_ext')
    tokenizer = BertTokenizer.from_pretrained(args.huggingface+'roberta_ext')
    model = NlpClassifierMultilabel(pretrained_model,38, 590, 10205)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)

    # read custom data
    data_files = {
        "train": args.data_input + "train.pqt",
        "test": args.data_input + "test.pqt"}
    raw_datasets = load_dataset("parquet", data_files=data_files)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # post process
    tokenized_datasets = tokenized_datasets.remove_columns(["title"])
    tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
    tokenized_datasets.set_format("torch")
    # sample weight
    samples_weight = get_class_weights(pd.read_parquet(data_files['train']))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, num_samples=len(samples_weight))

    #train params
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=False, pin_memory=True, drop_last=True, batch_size=batch_size, collate_fn=data_collator,
        sampler=sampler, num_workers=16
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], shuffle=True, pin_memory=True, drop_last=True, batch_size=batch_size, collate_fn=data_collator
    )
    num_training_steps = num_epochs * len(train_dataloader)
    # optimizer_emb = AdamW(model.emb_layer.parameters(), lr=5e-5)
    # lr_scheduler_emb = get_scheduler(
    #     name="linear", optimizer=optimizer_emb, num_warmup_steps=0, num_training_steps=num_training_steps
    # )

    optimizer_fc = AdamW(model.parameters(), lr=5e-5)
    lr_scheduler_fc = get_scheduler(
        name="linear", optimizer=optimizer_fc, num_warmup_steps=0.25*num_training_steps, num_training_steps=num_training_steps
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
            tag_labels=batch['tag_id']
            lv1_category_labels = batch['lv1_category_id']
            lv2_category_labels = batch['lv2_category_id']
            lv1_category_preds, lv2_category_preds, tag_preds = model(query_input_ids=batch['input_ids'],
                                                            query_token_type_ids=batch['token_type_ids'],
                                                            query_attention_mask=batch['attention_mask'],
                                                            firstcate_label=batch['lv1_category_id'],
                                                            secondcate_label=batch['lv2_category_id'],
                                                            tag_label=batch['tag_id'],
                                                            is_test=False)
            loss = args.lv1_weight * entroy(lv1_category_preds, lv1_category_labels) + \
                   args.lv2_weight * entroy(lv2_category_preds, lv2_category_labels) + \
                   args.tag_weight * entroy(tag_preds, tag_labels)
            loss.backward()
            tag_predictions = torch.argmax(tag_preds, dim=-1)
            train_accuracy(tag_predictions, tag_labels)


            # optimizer_emb.step()
            # lr_scheduler_emb.step()
            # optimizer_emb.zero_grad()

            optimizer_fc.step()
            lr_scheduler_fc.step()
            optimizer_fc.zero_grad()

            train_acc = train_accuracy.compute().cpu().detach().numpy()
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Acc/train', train_acc, global_step)

            progress_bar.set_postfix(loss=f'%.2f' % loss.cpu().detach().numpy(), acc=f'%.2f'%train_acc, test_acc=f'%.2f'%test_acc)
            progress_bar.update(1)
            global_step += 1
            if global_step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    for batch in test_dataloader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        lv1_category_preds, lv2_category_preds, tag_preds = model(query_input_ids=batch['input_ids'],
                                          query_token_type_ids=batch['token_type_ids'],
                                          query_attention_mask=batch['attention_mask'],
                                          firstcate_label=batch['lv1_category_id'],
                                          secondcate_label=batch['lv2_category_id'],
                                          tag_label=batch['tag_id'],
                                          is_test=True)
                        test_tag_labels = batch['tag_id']
                        test_tag_predictions = torch.argmax(tag_preds, dim=-1)
                        test_accuracy(test_tag_predictions, test_tag_labels)
                    test_acc = test_accuracy.compute().cpu().detach().numpy()
                    writer.add_scalar('Acc/test', test_acc, global_step)
            # save model
            if global_step % 1000 == 0:
                torch.save(model, args.data_output + '{}.pt'.format(global_step))
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
