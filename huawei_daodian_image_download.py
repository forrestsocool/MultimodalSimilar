#-*- coding: utf-8 -*-
import os
import os
import requests
import pandas as pd
import numpy as np
import cv2
import requests
import argparse
import os
import sys
import threadpool
from tqdm.auto import tqdm
import logging

show_download_status = True

def request_download(image_url, goodssku, num):
    image_path = '{}/{}/{}.jpg'.format(args.train_url, goodssku, num)
    if os.path.exists(image_path):
        return
    os.system('mkdir -p {}/{}'.format(args.train_url, goodssku))
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
        logging.error(e)
        #print("{}, {},  {} , {}".format(item['spu_sn'], item['spu_name'], item['ad_mediaurl'], item['detail_mediaurl']))


parser = argparse.ArgumentParser()
# parser.add_argument("--dt", type=str)
parser.add_argument('--train_url', type=str, default='obs://xsyx-modelarts-gz/lisensen/goodssku_image', help='test')
args = parser.parse_args()

if __name__ == '__main__':

    pd_frxs_skusn = pd.read_csv('./pd_frxs_skusn_2.csv')
    print("pd_frxs_skusn cnt : {}".format(len(pd_frxs_skusn)),flush=True)

    pd_frxs_skusn = pd_frxs_skusn[pd_frxs_skusn['sku']>0]
    pd_frxs_skusn['sku'] = pd_frxs_skusn['sku'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')
    pd_frxs_skusn['img_id'] = pd_frxs_skusn['img_id'].apply(lambda x: str(int(x)) if not np.isnan(x) else '')

    print(f'pd_frxs_skusn length : {len(pd_frxs_skusn)}', flush=True)

    pool = threadpool.ThreadPool(20)  # 线程池设置,最多同时跑20个线程

    progress_bar = tqdm(range(len(pd_frxs_skusn)))
    for index, item in pd_frxs_skusn.iterrows():
        task = threadpool.makeRequests(download_item, [item])
        pool.putRequest(task[0])
    pool.wait()
    progress_bar.close()
    #sc.stop()