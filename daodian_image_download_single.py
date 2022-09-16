#-*- coding: utf-8 -*-
import os
import os
import requests
import pandas as pd
import numpy as np
import cv2
os.environ['JAVA_HOME']='/opt/Bigdata/client/JDK/jdk1.8.0_272'
os.environ['SPARK_HOME']='/opt/Bigdata/client/Spark2x/spark'
# 若下面两条命令已加入 bashrc，可注释掉
os.system('source /opt/Bigdata/client/bigdata_env')
os.system("source /opt/Bigdata/client/Hudi/component_env")
os.system('kinit -kt /opt/Bigdata/client/user.keytab GPSearch')
# os.system("source /opt/Bigdata/client/bigdata_env")
# os.system("/opt/Bigdata/client/KrbClient/kerberos/bin/kinit -kt /workspace/gpsearch.keytab GPSearch")
#export HADOOP_USER_NAME=hive
#os.environ['HADOOP_USER_NAME']='hive'
import requests
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
# import threadpool
from tqdm.auto import tqdm
# import logging
import datetime

show_download_status = True

def request_download(image_url, goodssku, num):
    image_path = './goodssku_image_2/{}/{}.jpg'.format(goodssku, num)
    if os.path.exists(image_path):
        return
    os.system('mkdir -p ./goodssku_image_2/{}'.format(goodssku))
    r = requests.get(image_url)
    with open(image_path, 'wb') as f:
        f.write(r.content)

def download_item(item):
    try:
        goodssku = item['sku']
        img_id = item['img_id']
        url = item['originalimgurl']
        request_download(url, goodssku, img_id)
        if show_download_status:
            progress_bar.update(1)
            progress_bar.set_description("Processing {}-th iteration".format(index+1))
    except Exception as e:
        print(e)
        # logging.error(e)
        #print("{}, {},  {} , {}".format(item['spu_sn'], item['spu_name'], item['ad_mediaurl'], item['detail_mediaurl']))

oneday=datetime.timedelta(days=1)
tomorrow=(datetime.date.today()+oneday).strftime("%Y-%m-%d")
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=str, default=f'{tomorrow}')
args = parser.parse_args()

if __name__ == '__main__':
    #Load data
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "daodian_image_download").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    pd_frxs_skusn = spark.sql(f"""
        select frxs_skusn.*, spusn_image.originalimgurl, 
            cast(img_id-1 as string) as img_id 
        from 
            dm_recommend.daily_recommend_frxs_skusn_details_di frxs_skusn
        left join (
            select 
                spusn,
                originalimgurl,
                row_number() over (partition by spusn order by id) as img_id
            from
                ods.frxs_promotion_db_t_activity_product_image_rt
            where imgtype in ('PRIMARY')
        ) spusn_image
        on frxs_skusn.spu_sn = spusn_image.spusn
        where frxs_skusn.dt='{args.dt}' and frxs_skusn.sku is not null
    """).toPandas()
    print("pd_frxs_skusn cnt : {}".format(len(pd_frxs_skusn)),flush=True)

    # pd_frxs_skusn = pd_frxs_skusn[pd_frxs_skusn['sku']>0]
    # pd_frxs_skusn['sku'] = pd_frxs_skusn['sku'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')
    # pd_frxs_skusn['img_id'] = pd_frxs_skusn['img_id'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')

    # print(f'pd_frxs_skusn length : {len(pd_frxs_skusn)}', flush=True)

    #pool = threadpool.ThreadPool(20)  # 线程池设置,最多同时跑20个线程

    progress_bar = tqdm(range(len(pd_frxs_skusn)))
    for index, item in pd_frxs_skusn.iterrows():
        download_item(item)
        progress_bar.update(1)
    # pool.wait()
    progress_bar.close()
    sc.stop()
    sys.exit()