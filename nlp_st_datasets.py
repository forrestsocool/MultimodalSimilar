import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import random


class NlpSTDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.tag_th = 0.7
        self.second_cate_th = 0.2
        self.first_cate_th = 0.1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        curr_tag_id = row.tag_id
        second_cate_id = row.lv2_category_id
        first_cate_id = row.lv1_category_id
        query = row.title
        sku_sn_name = row.sku_sn_name
        r = random.uniform(0,1.0)
        #generate positive sample
        final_result = {}
        title = None
        label = None
        if r > 0.5:
            rr = random.uniform(0,1.0)
            # same firstcate
            if rr < self.first_cate_th:
                pd_choose = self.df[(self.df['lv1_category_id']==first_cate_id) & (self.df['sku_sn_name'] != sku_sn_name)]
                if len(pd_choose) >= 1:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title = pd_choose['title'][0]
            # same secondcate
            elif rr < self.second_cate_th or curr_tag_id == -1:
                pd_choose = self.df[(self.df['lv2_category_id']==second_cate_id) & (self.df['sku_sn_name'] != sku_sn_name)]
                if len(pd_choose) >= 1:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title =  pd_choose['title'][0]
            # same tag
            elif rr < self.tag_th:
                pd_choose = self.df[(self.df['tag_id']==curr_tag_id) & (self.df['sku_sn_name'] != sku_sn_name)]
                if len(pd_choose) > 2:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title = pd_choose['title'][0]
            label = torch.tensor(1.0).long()
        # generate negative sample
        else:
            title = None
            rr = random.uniform(0, 1.0)
            # diff tag same second cate
            if rr < self.first_cate_th and curr_tag_id != -1:
                pd_choose = self.df[
                    (self.df['tag_id'] != curr_tag_id) &
                    (self.df['lv2_category_id'] == second_cate_id) &
                    (self.df['title'] != query)]
                if len(pd_choose) >= 1:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title = pd_choose['title'][0]
            # diff second cate same first cate
            elif rr < self.second_cate_th:
                pd_choose = self.df[
                    (self.df['lv1_category_id'] == first_cate_id) &
                    (self.df['lv2_category_id'] != second_cate_id) &
                    (self.df['title'] != query)]
                if len(pd_choose) >= 1:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title = pd_choose['title'][0]
            # diff first cate
            elif rr < self.tag_th:
                pd_choose = self.df[
                    (self.df['lv1_category_id'] != first_cate_id) &
                    (self.df['title'] != query)]
                if len(pd_choose) >= 1:
                    pd_choose = pd_choose.sample(n=1, random_state=42).reset_index()
                    title = pd_choose['title'][0]
            label = torch.tensor(0.0).long()

        if title == None:
            title = query
            label = torch.tensor(1.0).long()
        query_tokens = self.transform(query)
        title_tokens = self.transform(title)
            # final_result['query_input_ids'] = query_tokens['input_ids']
            # final_result['query_attention_mask'] = query_tokens['attention_mask']
            # final_result['query_token_type_ids'] = query_tokens['token_type_ids']
            # final_result['title_input_ids'] = title_tokens['input_ids']
            # final_result['title_attention_mask'] = title_tokens['attention_mask']
            # final_result['title_token_type_ids'] = title_tokens['token_type_ids']
        return query_tokens,title_tokens,label