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

def get_emb_cv(frxs_product):
    pd_result = pd.DataFrame(columns=['goods_sku','cv_emb'])
    pd_spusn_proceed = frxs_product
    progress_bar = tqdm(range(len(pd_spusn_proceed)))
    tmp_list = []
    for i, item in pd_spusn_proceed.iterrows():
        progress_bar.update(1)
        try:
            goods_sku = item['goods_sku']
            emb_path = '/opt/pipeline/recommend/MultimodalSimilar/goodssku_image_2/{}/emb.txt'.format(goods_sku)
            if os.path.exists(emb_path):
                embedding = np.loadtxt(emb_path).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)
                embedding_str = ','.join(str(s) for s in embedding)
                embedding_str = f'[{embedding_str}]'
                #result[goods_sku] = embedding_str
                tmp_list.append({'goods_sku': goods_sku, 'cv_emb':embedding_str})
                if len(tmp_list) > 1000:
                    pd_result = pd_result.append(tmp_list, ignore_index=True)
                    tmp_list = []
            else:
                continue
        except Exception as e:
            print(e)
    if len(tmp_list) > 0:
        pd_result = pd_result.append(tmp_list, ignore_index=True)
    progress_bar.close()
    return pd_result

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    #Load data
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "goodssku_emb_cv").set("spark.ui.port", "4060")
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
    result = get_emb_cv(frxs_product)
    result['modifydate'] = args.dt
    # result = pd.read_parquet('./goodssku_emb.pqt')
    print(result.columns)
    schema = StructType([
        StructField("goods_sku", StringType(), False),
        StructField("cv_emb", StringType(), True),
        StructField("modifydate", StringType(), True)
    ])
    print(len(result))
    df_result = spark.createDataFrame(result, schema)
    # print(df_result.count())
    df_result = df_result.repartition(3000)
    df_result.repartition(3000).write.mode('overwrite').saveAsTable("tmp.tmp_goodssku_embedding_cv")
    spark.sql("""
        insert overwrite table dm_recommend.goodssku_embedding_cv
        select * from tmp.tmp_goodssku_embedding_cv
    """)
    sc.stop()
