#-*- coding: utf-8 -*-
import os
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
        text=preprocess_for_infer([example["spu_name"]])[0],
        padding="max_length",
        max_length=128,
        truncation=True)
    # img_path = "/opt/pipeline/recommend/MultimodalSimilar/yuanben_spusn_info/spusn_image/{}.jpg".format(example["spu_sn"])
    # img = Image.open(img_path).convert('RGB')
    # img_tensor = transform_eff(img)  # transform and add batch dimension
    # return img_tensor, title_tokens
    return title_tokens

config = {'input_size': (3, 320, 320),
         'interpolation': 'bicubic',
         'mean': (0.485, 0.456, 0.406),
         'std': (0.229, 0.224, 0.225),
         'crop_pct': 1.0}
transform_eff = create_transform(**config)
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

parser = argparse.ArgumentParser()
parser.add_argument("--redis_host", type=str, default='1.1.1.1')
parser.add_argument("--redis_password", type=str, default='password')
parser.add_argument("--redis_port", type=int, default=6379)
parser.add_argument("--redis_db", type=int, default=15)
parser.add_argument("--exp_seconds", type=int, default=7*24*3600)
args = parser.parse_args()

if __name__ == '__main__':
    #load data
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "yuanben_multimodal_similar").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    pd_multimodal = spark.sql("""
        select distinct spu_sn, spu_name
        from dm_recommend.dws_recommend_dj_frxs_skusn_details_di 
        where status = 'UP'
    """).toPandas()
    print("pd_multimodal cnt : {}".format(len(pd_multimodal)),flush=True)
    sc.stop()
    #load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torch.load('./nlp_model/4000.pt', map_location=device)
    model.eval()
    vec_result = []
    spusn_list = []
    print("calc embedding...")
    progress_bar = tqdm(range(len(pd_multimodal)))
    for i, item in pd_multimodal.iterrows():
        try:
            title_tokens = tokenize_function(item)
            title_tensors = data_collator(title_tokens)
            #ticks = time.time()
            embedding = model.predict_emb(
                            query_input_ids=title_tensors['input_ids'].unsqueeze(0).to(device),
                            query_token_type_ids=title_tensors['token_type_ids'].unsqueeze(0).to(device),
                            query_attention_mask=title_tensors['attention_mask'].unsqueeze(0).to(device))
            vec_result.append(np.array(embedding.cpu().detach())[0])
            spusn_list.append(item['spu_sn'])
            progress_bar.update(1)
        except:
            print("error processing {}".format(item['spu_sn']))
    mkl.get_max_threads()
    d = 768
    th_score = 0.9

    vec_result_np = np.asarray(vec_result)
    normalize_L2(vec_result_np)
    #index = faiss.IndexFlatL2(d)  # build the index
    #print(index.is_trained)  # 表示索引是否需要训练的布尔值
    index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)  # build the index
    index.add(vec_result_np)  # add vectors to the index
    print(index.ntotal)
    D, I = index.search(vec_result_np, 13)  # actual search
    print("emb similar output cnt : {}".format(len(I)),flush=True)
    pool = redis.ConnectionPool(host=args.redis_host, port=args.redis_port, password=args.redis_password, db=args.redis_db)
    print("redis: {}:{} {} {}".format(args.redis_host, args.redis_port, args.redis_password, args.redis_db),flush=True)
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline()  # 创建一个管道
    for i in range(0, len(I)):
        curr_spusn = spusn_list[i]
        curr_spusn_similar_str = ''
        for similar_index, score in zip(I[i][1:],D[i][1:]):
            if score > th_score:
                curr_spusn_similar_str += spusn_list[similar_index] + ','
        curr_spusn_similar_str = curr_spusn_similar_str.strip(',')
        if curr_spusn_similar_str=='':
            print('continue',flush=True)
            continue
        else:
            print(curr_spusn_similar_str,flush=True)
            pipe.set('dj_similar:{}'.format(curr_spusn), curr_spusn_similar_str)
            pipe.expire('dj_similar:{}'.format(curr_spusn) , args.exp_seconds)
    pipe.execute()