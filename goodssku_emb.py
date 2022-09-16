#-*- coding: utf-8 -*-
import os
# import cv2
os.environ['JAVA_HOME']='/opt/Bigdata/client/JDK/jdk1.8.0_272'
os.environ['SPARK_HOME']='/opt/Bigdata/client/Spark2x/spark'
# 若下面两条命令已加入 bashrc，可注释掉
# os.system('source /opt/Bigdata/client/bigdata_env')
# os.system('kinit -kt /opt/Bigdata/client/user.keytab GPSearch')
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
from pyspark.sql import SQLContext
import pandas as pd
from transformers import BertTokenizer, BertModel, DataCollatorWithPadding
import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import fasttext
from string import digits
from nlp_classifier_multilabel import NlpClassifierMultilabel

fasttext_result = pd.read_csv('fasttext_result.csv', delimiter='\t')
fasttext_result[["goods_sku"]] = fasttext_result[["goods_sku"]].astype(str)
bert_result = pd.read_csv('bert_result.csv', delimiter='\t')
bert_result[["goods_sku"]] = bert_result[["goods_sku"]].astype(str)
fasttext_result_keys = fasttext_result['goods_sku'].unique()
print(fasttext_result_keys)
bert_result_keys = bert_result['goods_sku'].unique()

fasttext_result_keys=[]
bert_result_keys=[]

#
# print(bert_result.columns)
# print(bert_result_keys[0])
# cursku = bert_result_keys[0]
# print(bert_result[bert_result['goods_sku']==cursku]['bert_emb'][0])

table = str.maketrans('', '', digits)
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def tokenize_function(example):
    return tokenizer(text=example, padding="max_length", max_length=80, truncation=True)

def gen_title(item):
    sku_sn_name = item['product_name']
    lv1_category_name = item['first_level_category_name'].translate(table)
    lv2_category_name = item['second_level_category_name'].translate(table)
    goods_title = item['product_title'].translate(table) if type(item['product_title']) == type('a') else ''
    title = '{} {} {} {}'.format(lv1_category_name, lv2_category_name, sku_sn_name, goods_title)
    title = ' '.join(title.split())
    title = title.strip()
    return title

def get_emb_fasttext(fasttext_result, frxs_product, nlp_model):
    # result = pd.DataFrame(columns=['goods_sku','fasttext_emb'])
    pd_spusn_proceed = frxs_product
    progress_bar = tqdm(range(len(pd_spusn_proceed)))
    for i, item in pd_spusn_proceed.iterrows():
        progress_bar.update(1)
        try:
            goods_sku = item['goods_sku']
            if goods_sku not in fasttext_result_keys:
                embedding = nlp_model.get_sentence_vector(item['title'])
                embedding_str = ','.join(str(s) for s in embedding)
                fasttext_result = fasttext_result.append({'goods_sku': goods_sku, 'fasttext_emb': embedding_str}, ignore_index=True)
        except Exception as e:
            print(e)
    return fasttext_result

def get_emb_bert(bert_result, frxs_product, nlp_model):
    # result = pd.DataFrame(columns=['goods_sku','bert_emb'])
    pd_spusn_proceed = frxs_product
    progress_bar = tqdm(range(len(pd_spusn_proceed)))
    for i, item in pd_spusn_proceed.iterrows():
        progress_bar.update(1)
        try:
            goods_sku = item['goods_sku']
            if goods_sku not in bert_result_keys:
                title_tokens = tokenize_function(item['title'])
                title_tensors = data_collator([title_tokens])
                embedding = nlp_model.predict_emb(
                    query_input_ids=title_tensors['input_ids'].to(device),
                    query_token_type_ids=title_tensors['token_type_ids'].to(device),
                    query_attention_mask=title_tensors['attention_mask'].to(device))
                embedding_np = embedding.cpu().detach().numpy()[0]
                embedding_str = ','.join(str(s) for s in embedding_np)
                bert_result = bert_result.append({'goods_sku': goods_sku, 'bert_emb':embedding_str}, ignore_index=True)
        except Exception as e:
            print(e)
    return bert_result

def get_emb_cv(frxs_product):
    pd_result = pd.DataFrame(columns=['goods_sku','cv_emb'])
    pd_spusn_proceed = frxs_product
    progress_bar = tqdm(range(len(pd_spusn_proceed)))
    for i, item in pd_spusn_proceed.iterrows():
        progress_bar.update(1)
        try:
            goods_sku = item['goods_sku']
            emb_path = '/opt/pipeline/recommend/MultimodalSimilar/goodssku_image_2/{}/emb.txt'.format(goods_sku)
            if os.path.exists(emb_path):
                embedding = np.loadtxt(emb_path).astype('float32')
                embedding_str = ','.join(str(s) for s in embedding)
                #result[goods_sku] = embedding_str
                pd_result = pd_result.append({'goods_sku': goods_sku, 'cv_emb':embedding_str}, ignore_index=True)
            else:
                continue
        except Exception as e:
            print(e)
    return pd_result

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    #Load data
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "goodssku_emb_calc").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    frxs_product = spark.sql(f"""
        select
            cast (goods_sku as string) as goods_sku,
            max(sku_sn_name) as product_name,
            max(goods_title) as product_title,
            max(lv1_category_name) as first_level_category_name,
            max(lv2_category_name) as second_level_category_name
        from
            dim.dim_sku_sn
        where dt='{args.dt}'
        group by goods_sku
    """).toPandas()
    frxs_product['title'] = frxs_product.apply(lambda x: gen_title(x), axis=1)

    frxs_product.to_parquet('frxs_product_goodssku.pqt')

    # frxs_product = pd.read_parquet('goodssku_info.pqt')

    #load fasttext model
    fasttext_model = fasttext.load_model('model_file_0804.bin')

    #load nlp model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nlp_model = torch.load('./nlp_model_v3/6000.pt', map_location=device)
    nlp_model = nlp_model.module
    nlp_model.to(device)
    nlp_model.eval()

    fasttext_result = get_emb_fasttext(fasttext_result,frxs_product, fasttext_model)
    bert_result = get_emb_bert(bert_result, frxs_product, nlp_model)
    cv_result = get_emb_cv(frxs_product)

    result = pd.merge(fasttext_result, bert_result, how='outer', on=['goods_sku'])
    result = pd.merge(result, cv_result, how='outer', on=['goods_sku'])
    result = result.fillna('')

    result['modifydate'] = args.dt
    # result = pd.read_parquet('./goodssku_emb.pqt')
    print(result.columns)
    schema = StructType([
        StructField("goods_sku", StringType(), False),
        StructField("fasttext_emb", StringType(), True),
        StructField("bert_emb", StringType(), True),
        StructField("cv_emb", StringType(), True),
        StructField("modifydate", StringType(), True)
    ])
    #result.to_parquet('./goodssku_emb.pqt')
    print(len(result))
    df_result = spark.createDataFrame(result, schema)
    # print(df_result.count())
    df_result = df_result.repartition(300)
    df_result.repartition(300).write.mode('overwrite').saveAsTable("dm_recommend.goodssku_embedding")

    #df.repartition(800).write.mode('overwrite').format("hive").saveAsTable("dm_recommend.goodssku_embedding")
    # df_result.registerTempTable('tmp_table')
    # hive_context = HiveContext(sc)
    # hive_context.sql(""""
    #     insert overwrite table dm_recommend.goodssku_embedding
    #     select * from tmp_table
    # """)