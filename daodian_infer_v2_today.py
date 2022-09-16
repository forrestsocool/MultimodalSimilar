#-*- coding: utf-8 -*-
# 版本更新：支持最近7天历史数据查询
import os
import cv2
os.environ['JAVA_HOME']='/opt/Bigdata/client/JDK/jdk1.8.0_272'
os.environ['SPARK_HOME']='/opt/Bigdata/client/Spark2x/spark'
# 若下面两条命令已加入 bashrc，可注释掉
os.system('source /opt/Bigdata/client/bigdata_env')
os.system('kinit -kt /opt/Bigdata/client/user.keytab GPSearch')
# os.system("source /opt/Bigdata/client/bigdata_env")
# os.system("/opt/Bigdata/client/KrbClient/kerberos/bin/kinit -kt /workspace/gpsearch.keytab GPSearch")
# os.system("source /opt/Bigdata/client/Hudi/component_env")
#export HADOOP_USER_NAME=hive
#os.environ['HADOOP_USER_NAME']='hive'
import requests
import pandas as pd
import traceback
import findspark
findspark.init()
import sys
# --jars hdfs:///user/lisensen/tools/jpmml-sparkml-executable-1.5.13.jar
# pyspark_submit_args = ' --executor-memory 2g --driver-memory 8g --executor-cores 2 --num-executors 30 --conf spark.shuffle.spill.numElementsForceSpillThreshold=2000000 --conf spark.memory.storageFraction=0.2 --conf spark.dlism=2000 --conf spark.sql.shuffle.partitions=2000 --conf spark.dynamicAllocation.enabled=false --conf spark.port.maxRetries=100 --conf spark.driver.maxResultSize=8g' + ' pyspark-shell'
pyspark_submit_args = ' --master local[*] --driver-memory 4g --executor-cores 2 --conf spark.driver.extraJavaOptions=" -Xss16384k" --conf spark.driver.memoryOverhead=4g --conf spark.local.dir=/opt/home/lisensen/temp --conf spark.shuffle.memoryFraction=0.1 --conf spark.kryoserializer.buffer.max=1800m' + ' pyspark-shell'
os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
os.environ['HADOOP_USER_NAME']='hdfs'
import argparse
import os
import sys
from pyspark import StorageLevel
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext,SQLContext,Row,SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf,col,column
import pyspark.sql.types as typ
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType, IntegerType
import torch
import pandas as pd
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from multimodal_classifier import MultimodalClassifier
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm.auto import tqdm
import time
import numpy as np
import mkl
import faiss
import redis
import argparse
from faiss import normalize_L2
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
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import json
import fasttext

ann_cnt_cv = 26
ann_cnt_nlp = 100
nlp_score_th = -0.6
cv_score_th = 0.15
area_list = [114, 102, 106, 112, 105, 101, 111, 104, 110, 108, 109, 107]
# area_list = [114]

stopwords=pd.read_csv("./stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values
remove_words = ['【福利秒杀】','【每日福利】','【福利爆款】','【专柜品质】','【1元秒杀】','【直播专用1元秒杀】','【','】','源本']

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


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

def getAugmentation(IMG_SIZE, isTraining=False):
    if isTraining:
        return albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.75),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(p=1.0)
        ])
    else:
        return albumentations.Compose([
            albumentations.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(p=1.0)
        ])

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

from string import digits
table = str.maketrans('', '', digits)

def gen_title(item):
    sku_sn_name = item['product_name']
    lv1_category_name = item['first_level_category_name'].translate(table)
    lv2_category_name = item['second_level_category_name'].translate(table)
    goods_title = item['product_title'].translate(table) if type(item['product_title']) == type('a') else ''
    title = '{} {} {} {}'.format(lv1_category_name, lv2_category_name, sku_sn_name, goods_title)
    title = ' '.join(title.split())
    title = title.strip()
    return title

# def gen_title(sku_sn_name,goods_title):
#     #title = '{} {} {} {}'.format(remove_num(lv1_category_name), remove_num(lv2_category_name), sku_sn_name, goods_title)
#     goods_title = str(goods_title) if str(goods_title) != 'nan' else ''
#     title = '{} {}'.format(sku_sn_name, goods_title)
#     title = title.strip()
#     return title

def tokenize_function(example):
    return tokenizer(text=preprocess_for_infer([example["title"]]), padding="max_length", max_length=128, truncation=True)

# def get_similar_nlp(frxs_product, nlp_model):
#     nlp_vec_result = []
#     nlp_spusn_list = []
#     sku_list = []
#     nlp_spuname_list = []
#     pd_spusn_proceed = frxs_product[['spu_sn', 'area_id', 'product_name', 'product_title', 'sku']]
#     pd_spusn_proceed['title'] = pd_spusn_proceed.apply(lambda x: gen_title(x), axis=1)
#     progress_bar = tqdm(range(len(pd_spusn_proceed)))
#     for i, item in pd_spusn_proceed.iterrows():
#         try:
#             title_tokens = tokenize_function(item)
#             # print(title_tokens)
#             title_tensors = data_collator(title_tokens)
#             # ticks = time.time()
#             embedding = nlp_model.predict_emb(
#                 query_input_ids=title_tensors['input_ids'].to(device),
#                 query_token_type_ids=title_tensors['token_type_ids'].to(device),
#                 query_attention_mask=title_tensors['attention_mask'].to(device))
#             # print(embedding)
#             # embedding = cv_model.predict_emb(img_tensor.unsqueeze(0).to(device))
#             nlp_vec_result.append(embedding.cpu().detach().numpy()[0])
#             progress_bar.update(1)
#             nlp_spusn_list.append(item['spu_sn'])
#             nlp_spuname_list.append(item['product_name'])
#             sku_list.append(item['sku'])
#         except Exception as e:
#             print(e)
#             print(item['spu_sn'])
#     arr_result = np.asarray(nlp_vec_result)
#     normalize_L2(arr_result)
#     mkl.get_max_threads()
#     d = 768
#     index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)  # build the index
#     index.add(arr_result)  # add vectors to the index
#     D, I = index.search(arr_result, ann_cnt_nlp)  # actual search
#     print("nlp emb similar output cnt : {}".format(len(I)), flush=True)
#     nlp_sku_map = {}
#     for i in range(len(nlp_spusn_list)):
#         spusn = nlp_spusn_list[i]
#         nlp_sku_map[spusn] = []
#         for similar_index, score in zip(I[i][1:], D[i][1:]):
#             if score > nlp_score_th:
#                 nlp_sku_map[spusn].append(nlp_spusn_list[similar_index])
#     progress_bar.close()
#     return nlp_sku_map

def get_similar_fasttext(frxs_product, nlp_model):
    nlp_vec_result = []
    nlp_spusn_list = []
    sku_list = []
    nlp_spuname_list = []
    cate_list = []
    pd_spusn_proceed = frxs_product
    progress_bar = tqdm(range(len(pd_spusn_proceed)))
    for i, item in pd_spusn_proceed.iterrows():
        try:
            embedding = nlp_model.get_sentence_vector(item['title'])
            nlp_vec_result.append(embedding)
            progress_bar.update(1)
            nlp_spusn_list.append(item['spu_sn'])
            nlp_spuname_list.append(item['product_name'])
            sku_list.append(item['sku'])
            cate_list.append(item['first_level_category_id'])
        except Exception as e:
            print(e)
            print(item['spu_sn'])
    arr_result = np.asarray(nlp_vec_result)
    normalize_L2(arr_result)
    mkl.get_max_threads()
    d = 100
    index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)  # build the index
    index.add(arr_result)  # add vectors to the index
    D, I = index.search(arr_result, len(arr_result))  # actual search
    print("nlp emb similar output cnt : {}".format(len(I)), flush=True)
    nlp_sku_map = {}
    for i in range(len(nlp_spusn_list)):
        spusn = nlp_spusn_list[i]
        nlp_sku_map[spusn] = []
        cate_id = cate_list[i]
        for similar_index, score in zip(I[i][1:], D[i][1:]):
            curr_cate_id = cate_list[similar_index]
            if score > nlp_score_th \
                and cate_id==curr_cate_id \
                and nlp_spusn_list[similar_index] != spusn \
                and nlp_spusn_list[similar_index] not in nlp_sku_map[spusn]:
                nlp_sku_map[spusn].append(nlp_spusn_list[similar_index])
            if len(nlp_sku_map[spusn]) > ann_cnt_nlp:
                break
    progress_bar.close()
    return nlp_sku_map

def get_similar_cv(frxs_product, cv_model):
    vec_result = []
    sku_image_list = []
    cate_list = []
    spusn_list = []
    progress_bar = tqdm(range(len(frxs_product)))
    validation_aug = getAugmentation(512, isTraining=False)
    for i, item in frxs_product.iterrows():
        goods_sku = str(int(item['sku']))
        path = '/opt/pipeline/recommend/MultimodalSimilar/goodssku_image_2/{}/0.jpg'.format(goods_sku)
        emb_path = '/opt/pipeline/recommend/MultimodalSimilar/goodssku_image_2/{}/emb.txt'.format(goods_sku)
        emb_exist = True if os.path.exists(emb_path) else False
        try:
            if not emb_exist:
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                aug = validation_aug(image=image)
                tensor = aug['image']
                embedding = cv_model.predict_emb(tensor.unsqueeze(0).to(device))
                image_cnt = 1
                for i in range(1, 8):
                    path_new = './goodssku_image_2/{}/{}.jpg'.format(goods_sku, i)
                    if os.path.exists(path_new):
                        image = cv2.imread(path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        aug = validation_aug(image=image)
                        tensor = aug['image']
                        embedding_new = cv_model.predict_emb(tensor.unsqueeze(0).to(device))
                        embedding = torch.add(embedding, embedding_new)
                        image_cnt += 1
                    else:
                        break
                embedding = torch.div(embedding, image_cnt)
                vec_result.append(embedding.cpu().detach().numpy()[0])
                np.savetxt(emb_path, embedding.cpu().detach().numpy()[0])
            else:
                vec_result.append(np.loadtxt(emb_path).astype('float32'))
            spusn_list.append(item['spu_sn'])
            sku_image_list.append(path)
            cate_list.append(item['second_level_category_id'])
        except Exception as e:
            print(e)
            print(path)
            # break
        progress_bar.update(1)
    arr_result = np.asarray(vec_result)
    normalize_L2(arr_result)
    mkl.get_max_threads()
    d = 512
    index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)  # build the index
    # print(index.is_trained)  # 表示索引是否需要训练的布尔值
    index.add(arr_result)  # add vectors to the index
    # print(index.ntotal)
    D, I = index.search(arr_result, ann_cnt_cv)  # actual search
    print("cv emb similar output cnt : {}".format(len(I)), flush=True)
    cv_sku_map = {}
    for i in range(len(spusn_list)):
        cate_id = cate_list[i]
        spusn = spusn_list[i]
        cv_sku_map[spusn] = []
        for similar_index, score in zip(I[i][1:], D[i][1:]):
            curr_cate = cate_list[similar_index]
            if score > cv_score_th \
                and cate_id==curr_cate \
                and spusn_list[similar_index] != spusn \
                and spusn_list[similar_index] not in cv_sku_map[spusn]:
                cv_sku_map[spusn].append(spusn_list[similar_index])
    progress_bar.close()
    return cv_sku_map

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=str)
parser.add_argument("--redis_host", type=str, default='1.1.1.1')
parser.add_argument("--redis_password", type=str, default='password')
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--redis_db", type=int, default=15)
parser.add_argument("--exp_seconds", type=int, default=int(1.5*24*3600))
args = parser.parse_args()


if __name__ == '__main__':
    #Load data
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "daodian_multimodal_similar").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    frxs_product = spark.sql(f"""
        select * from dm_recommend.daily_recommend_frxs_skusn_details_di
        where sku is not null 
        and sku != ''
        and dt='{args.dt}'
    """).toPandas()
    frxs_product['title'] = frxs_product.apply(lambda x: gen_title(x), axis=1)
    print("frxs_product cnt : {}".format(len(frxs_product)),flush=True)
    area_list = list(frxs_product['area_id'].unique())
    sc.stop()

    #Load Model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # nlp_model = torch.load('./nlp_model_daodian/18000.pt', map_location=device)
    # nlp_model.to(device)
    # nlp_model.eval()
    fasttext_model = fasttext.load_model('model_file_0804.bin')
    os.system("cp ./efficientnet_b4_ra2_320-7eb33cd5.pth /home/search/.cache/torch/hub/checkpoints/")
    cv_model = CvClassifier('efficientnet_b4', fc_dim=512, num_labels=4181)
    cv_model.load_state_dict(torch.load('./cv_model_27_2022-07-21_softmax_512x512_tf_efficientnet_b4.pt', map_location=device))
    cv_model.to(device)
    cv_model.eval()

    #Calc Part
    merged_result = {}
    for area_id in area_list:
        print(f'processing area: {area_id}',flush=True)
        merged_result[area_id] = {}
        frxs_product_area=frxs_product[frxs_product['area_id']==area_id]
        #nlp_similar_map = get_similar_nlp(frxs_product_area, nlp_model)
        nlp_similar_map = get_similar_fasttext(frxs_product_area, fasttext_model)
        cv_similar_map = get_similar_cv(frxs_product_area, cv_model)
        for k in cv_similar_map.keys():
            similar_result = []
            similar_result.extend(spusn for spusn in cv_similar_map[k])
            similar_result.extend(spusn for spusn in nlp_similar_map[k] if spusn not in similar_result)
            merged_result[area_id][k]=similar_result
        for k in nlp_similar_map.keys():
            if k not in merged_result[area_id]:
                merged_result[area_id][k] = nlp_similar_map[k]

    print(f'merged_result length : {len(merged_result.keys())}',flush=True)

    pool = redis.ConnectionPool(host=args.redis_host, port=args.redis_port, password=args.redis_password, db=args.redis_db)
    print("redis: {}:{} {} {}".format(args.redis_host, args.redis_port, args.redis_password, args.redis_db),flush=True)
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline()  # 创建一个管道
    for area_id in area_list:
        area_results = merged_result[area_id]
        for key in area_results:
            result = area_results[key]
            target_dt_aka = args.dt.replace('-', '')
            write_key = f'{target_dt_aka}:{key}'
            if len(result) > 0:
                result_str = json.dumps(result).replace('[', '').replace(']', '').replace('"', '').replace(' ', '')
                pipe.set(write_key, result_str)
                pipe.expire(write_key, args.exp_seconds)
        pipe.execute()
        print(f'area {area_id} process finish', flush=True)
