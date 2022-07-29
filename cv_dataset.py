import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import math
import os
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


class CvDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get row at index idx
        #         print("idx",idx)

        row = self.df.iloc[idx]
        #         print(row)
        label = row.tag_new_id
        image_path = os.path.join(self.root_dir, r'{}.jpg'.format(row.goods_sku))

        # read image convert to RGB and apply augmentation
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                aug = self.transform(image=image)
                image = aug['image']
        except:
            print(image_path)
            return None

        return image, torch.tensor(label).long()
