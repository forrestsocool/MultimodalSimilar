import sys
import os
import torch
import torch.nn as nn
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
import torchvision
from cv_classifier import CvClassifier
from timm.data import ImageDataset, IterableImageDataset, AugMixDataset, create_loader, create_dataset
import timm.optim
import timm.scheduler
from tqdm.auto import tqdm
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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
import numpy as np
import pandas as pd
import re
from torch.utils.tensorboard import SummaryWriter
from multimodal_dataset import MultimodalDataset
from multimodal_classifier import MultimodalClassifier

batch_size = 48
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
    title_tokens = tokenizer(
        text=preprocess_for_infer(example["spu_name"]),
        padding="max_length",
        max_length=128,
        truncation=True)
    img_path = "/home/ma-user/work/MultimodalSimilar/yuanben_spusn_20220623/spusn_image/{}.jpg".format(example["spu_sn"])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform_eff(img)  # transform and add batch dimension
    return img_tensor, title_tokens

# def compute_metrics(logits, labels):
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# load model
img_model = torch.load('/data/result/multimodal/cv_model/44000.pt.bak')
config = resolve_data_config({}, model=img_model.ptm)
transform_eff = create_transform(**config)
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def collate_fn(batch):
    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number
    tensors, targets = [], []
    img_tensors = []
    token_results = []
    # Gather in lists, and encode labels as indices
    for img_tensor, token_result, label in batch:
        img_tensors += [img_tensor]
        targets += [label]
        token_results += [token_result]

    # Group the list of tensors into a batched tensor
    tensors = data_collator(token_results)
    tensors['img_tensor'] = torch.stack(img_tensors)
    tensors['labels'] = torch.stack(targets)
    #targets = torch.stack(targets)

    #return tensors, targets
    return tensors

# def collate_fn_test(batch):
#     # A data tuple has the form:
#     # waveform, sample_rate, label, speaker_id, utterance_number
#
#     tensors = []
#     img_tensors = []
#     token_results = []
#     # Gather in lists, and encode labels as indices
#     for img_tensor, token_result in batch:
#         img_tensors += [img_tensor]
#         token_results += [token_result]
#
#     # Group the list of tensors into a batched tensor
#     tensors = data_collator(token_results)
#     tensors['img_tensor'] = torch.stack(img_tensors)
#
#     return tensors

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter()

    model = MultimodalClassifier(device = device,
                              cv_classifier_path = '/data/result/multimodal/cv_model/bak_44000.pt',
                              nlp_classifier_path = '/data/result/multimodal/nlp_model/2000.pt',
                              emb_size = 2560,
                              num_labels = 796)

    # read custom data
    train_dataset = MultimodalDataset(
        tokenizer=tokenizer,
        transform=transform_eff,
        csv_path='/data/sample/yuanben_spusn_20220623/train_multimodal.csv',
        img_path='/data/sample/yuanben_spusn_20220623/spusn_image/',
        use_label=True)
    test_dataset = MultimodalDataset(
        tokenizer=tokenizer,
        transform=transform_eff,
        csv_path='/data/sample/yuanben_spusn_20220623/test_multimodal.csv',
        img_path='/data/sample/yuanben_spusn_20220623/spusn_image/',
        use_label=True)
    train_dataloader = DataLoader(
        dataset=train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=16
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn, num_workers=16
    )

    num_training_steps = num_epochs * len(train_dataloader)

    #different layers use different learning rate
    optimizer_emb = AdamW([
        {'params': model.cv.parameters()},
        {'params': model.nlp.parameters()},
        # {'params': model.classifier.parameters(), 'lr': 1e-2}
    ], lr=5e-5)
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
            preds = model(img_input=batch['img_tensor'],
                        query_input_ids=batch['input_ids'],
                        query_token_type_ids=batch['token_type_ids'],
                        query_attention_mask=batch['attention_mask'],
                        label=batch['labels'])
            labels=batch['labels']
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
            if global_step % 1000 == 0:
                model.eval()
                for batch in test_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        preds = model(img_input=batch['img_tensor'],
                                      query_input_ids=batch['input_ids'],
                                      query_token_type_ids=batch['token_type_ids'],
                                      query_attention_mask=batch['attention_mask'],
                                      label=batch['labels'],
                                      is_test=True)
                    ground_truth = batch['labels']
                    predictions = torch.argmax(preds, dim=-1)
                    test_accuracy(predictions, ground_truth)
                    test_acc = test_accuracy.compute().cpu().detach().numpy()
                writer.add_scalar('Acc/test', test_acc, global_step)
                # save model
                torch.save(model,'./multimodal_model/{}.pt'.format(global_step))

