#-*- coding: utf-8 -*-
import os
os.environ['JAVA_HOME']='/opt/Bigdata/client/JDK/jdk1.8.0_272'
os.environ['SPARK_HOME']='/opt/Bigdata/client/Spark2x/spark'
os.system("source /opt/Bigdata/client/bigdata_env")
os.system("/opt/Bigdata/client/KrbClient/kerberos/bin/kinit -kt /workspace/gpsearch.keytab GPSearch")
os.system("source /opt/Bigdata/client/Hudi/component_env")
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


if not os.path.exists("./yuanben_spusn_info/spusn_image"):
    os.mkdir("./yuanben_spusn_info/spusn_image")
def request_download(image_url, spu_sn):
    image_path = './yuanben_spusn_info/spusn_image/{}.jpg'.format(spu_sn)
    if os.path.exists(image_path):
        return
    r = requests.get(image_url)
    with open('./yuanben_spusn_info/spusn_image/{}.jpg'.format(spu_sn), 'wb') as f:
        f.write(r.content)

if __name__ == '__main__':
    sparkConf = SparkConf()
    sparkConf.set("spark.app.name", "yuanben_media_download").set("spark.ui.port", "4060")
    spark = SparkSession.builder.config(conf=sparkConf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext

    pd_spu = spark.sql("""
        select distinct a.spusn, b.mediaurl
        from(
          select spusn, max(id) as id from
          ods.frxs_promotion_db_t_home_activity_product_sku_media_ro
          where 
          --datastatus = 'NORMAL' and 
          mediatype  = 'AD'
          group by spusn
        ) a
        left join ods.frxs_promotion_db_t_home_activity_product_sku_media_ro b
        on a.spusn=b.spusn and a.id=b.id
    """).toPandas()
    print("pd_spu cnt : {}".format(len(pd_spu)),flush=True)

    nums = 0
    for index, item in pd_spu.iterrows():
        nums += 1
        try:
            #ad_url = item['ad_mediaurl']
            #detail_url = item['detail_mediaurl']
            #url = ad_url if ad_url is not None else detail_url
            url = item['mediaurl']
            #print(url)
            request_download(url, item['spusn'])
            if nums % 100 == 0:
                print("proceed {}".format(nums))
        except Exception as e:
            print(e)
            break
    sc.stop()
