from torch.utils.data import Dataset
import pandas as pd
from pandas import read_parquet, read_csv
import pyarrow.parquet as pq
import torch
import logging
import multiprocessing
import time
import gc
import numpy as np
import re
from PIL import Image



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


class MultimodalDataset(Dataset):
    def __init__(self, tokenizer, transform, csv_path, img_path, use_label=False):
        self.dataframe = read_csv(csv_path)
        self.csv_path = csv_path
        self.img_path = img_path
        self.tokenizer = tokenizer
        self.use_label=use_label
        self.transform = transform

    def tokenize_function(self, spu_name):
        title_tokens = self.tokenizer(
            text=preprocess_for_infer([spu_name])[0],
            padding="max_length",
            max_length=128,
            truncation=True)
        return title_tokens

    def __getitem__(self, index):
        spusn = self.dataframe['spu_sn'][index]
        spu_name = self.dataframe['spu_name'][index]
        img_path = "{}/{}.jpg".format(self.img_path, spusn)
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        token_result = self.tokenize_function(spu_name)
        if self.use_label:
            label = torch.tensor(self.dataframe['cateid'][index].astype(int), dtype=torch.int64)
            return img_tensor, token_result, label
        else:
            return img_tensor, token_result

    def __len__(self):
        return len(self.dataframe)