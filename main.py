# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torch
import transformers
from transformers import BertTokenizer, BertModel

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # model = timm.create_model('efficientnet_b0', pretrained=True)
    # model.eval()
    #
    # config = resolve_data_config({}, model=model)
    # transform = create_transform(**config)
    tokenizer = BertTokenizer.from_pretrained('chinese-roberta-wwm-ext')
    model = BertModel.from_pretrained('chinese-roberta-wwm-ext')

    # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    # urllib.request.urlretrieve(url, filename)
    # img = Image.open(filename).convert('RGB')
    # tensor = transform(img).unsqueeze(0)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
