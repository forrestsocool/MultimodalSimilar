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
from nlp_sentence_transformer import NlpSentenceTransformer
from transformers import get_scheduler
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import re
from torch.utils.tensorboard import SummaryWriter
from nlp_st_datasets import NlpSTDataset
import torchmetrics

pretrained_model=None
tokenizer = None
model = None
batch_size = 200
num_epochs = 30
#
# stopwords=pd.read_csv("./stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
# stopwords=stopwords['stopword'].values
# remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']

#分词去停用词，并整理为fasttext要求的文本格式
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


from string import digits
table = str.maketrans('', '', digits)

def gen_title(item):
    sku_sn_name = item['sku_sn_name']
    lv1_category_name = item['lv1_category_name'].translate(table)
    lv2_category_name = item['lv2_category_name'].translate(table)
    goods_title = item['goods_title'].translate(table) if item['goods_title'] != None else ''
    title = '{} {} {} {}'.format(lv1_category_name, lv2_category_name, sku_sn_name, goods_title)
    title = ' '.join(title.split())
    title = title.strip()
    return title

# def c
# def tokenize_function(example):
#     return tokenizer(text=example, padding="max_length", max_length=80, truncation=True)ompute_metrics(logits, labels):
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors = {}
    targets = []
    query_token_results, title_token_results = [],[]
    # Gather in lists, and encode labels as indices
    for query, title, label in batch:
        query_token_results += [query]
        title_token_results += [title]
        targets += [label]

    # Group the list of tensors into a batched tensor
    query_tensors = data_collator(query_token_results)
    title_tensors = data_collator(title_token_results)
#     tensors['img_tensor'] = torch.stack(img_tensors)

    tensors['query_input_ids'] = query_tensors['input_ids']
    tensors['query_token_type_ids'] = query_tensors['token_type_ids']
    tensors['query_attention_mask'] = query_tensors['attention_mask']
    tensors['title_input_ids'] = title_tensors['input_ids']
    tensors['title_token_type_ids'] = title_tensors['token_type_ids']
    tensors['title_attention_mask'] = title_tensors['attention_mask']
    tensors['labels'] = torch.stack(targets)
    #targets = torch.stack(targets)

    #return tensors, targets
    return tensors

def get_class_weights(data):
    weight_dict=dict()
    # Format of row : PostingId, Image, ImageHash, Title, LabelGroup
    # LabelGroup index is 4 and it is representating class information
    for i, row in data.iterrows():
        weight_dict[row['tag_id']]=0
    # Word dictionary keys will be label and value will be frequency of label in dataset
    for i, row in data.iterrows():
        weight_dict[row['tag_id']]+=1
    # for each data point get label count data
    class_sample_count= np.array([weight_dict[row['tag_id']] for i, row in data.iterrows()])
    # each data point weight will be inverse of frequency
    weight = 1. / class_sample_count
    weight=torch.from_numpy(weight)
    return weight

if __name__ == '__main__':
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()
    # load pretrained model
    pretrained_model = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    model = NlpSentenceTransformer(pretrained_model)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    trainset = NlpSTDataset(pd.read_parquet('./nlp_st_train.pqt'),tokenize_function)
    validset = NlpSTDataset(pd.read_parquet('./nlp_st_test.pqt'),tokenize_function)

    samples_weight = get_class_weights(pd.read_parquet('./nlp_st_train.pqt'))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, num_samples=len(samples_weight))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # #train params
    train_dataloader = DataLoader(
        trainset, shuffle=False, drop_last=True, pin_memory=True, sampler=sampler,
        batch_size=batch_size, collate_fn=collate_fn, num_workers=16
    )
    test_dataloader = DataLoader(
        validset, shuffle=True, drop_last=True, pin_memory=True,
        batch_size=batch_size, collate_fn=collate_fn, num_workers=16
    )

    num_training_steps = num_epochs * len(train_dataloader)
    # optimizer_emb = AdamW(model.emb_layer.parameters(), lr=5e-5)
    # lr_scheduler_emb = get_scheduler(
    #     name="linear", optimizer=optimizer_emb, num_warmup_steps=0, num_training_steps=num_training_steps
    # )

    optimizer = AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0.25*num_training_steps, num_training_steps=num_training_steps
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
            preds = model(query_input_ids=batch['query_input_ids'],
                        query_token_type_ids=batch['query_token_type_ids'],
                        query_attention_mask=batch['query_attention_mask'],
                        title_input_ids=batch['title_input_ids'],
                        title_token_type_ids=batch['title_token_type_ids'],
                        title_attention_mask=batch['title_attention_mask'])
            loss = entroy(preds, labels)
            loss.backward()

            predictions = torch.argmax(preds, dim=-1)
            train_accuracy(predictions, labels)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_acc = train_accuracy.compute().cpu().detach().numpy()
            writer.add_scalar('Loss/train', loss.cpu().detach().numpy(), global_step)
            writer.add_scalar('Acc/train', train_acc, global_step)

            progress_bar.set_postfix(loss=f'%.2f' % loss.cpu().detach().numpy(), acc=f'%.2f'%train_acc, test_acc=f'%.2f'%test_acc)
            progress_bar.update(1)
            global_step += 1
            if global_step % 1000 == 0:
                model.eval()
                for batch in test_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        preds = model(query_input_ids=batch['query_input_ids'],
                                        query_token_type_ids=batch['query_token_type_ids'],
                                        query_attention_mask=batch['query_attention_mask'],
                                        title_input_ids=batch['title_input_ids'],
                                        title_token_type_ids=batch['title_token_type_ids'],
                                        title_attention_mask=batch['title_attention_mask'])
                        test_labels = batch['labels']
                        predictions = torch.argmax(preds, dim=-1)
                        test_accuracy(predictions, test_labels)
                    print('batch finish', flush=True)
                test_acc = test_accuracy.compute().cpu().detach().numpy()
                writer.add_scalar('Acc/test', test_acc, global_step)
            # save model
            #if global_step % 1000 == 0:
                torch.save(model,'./nlp_model_v2/{}.pt'.format(global_step))


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
