import redis
import pandas as pd

src_redis = redis.Redis(host='r-wz9mqqj0xl303rxf4f.redis.rds.aliyuncs.com', port=6379, db=5, password='EX=sal=2033-Od')
target_redis = redis.Redis(host='r-wz9mqqj0xl303rxf4f.redis.rds.aliyuncs.com', port=6379, db=16, password='EX=sal=2033-Od')

cnt = 0
pd_skusn = pd.read_csv('./pd_frxs_skusn_2.csv')
# for i,item in pd_skusn.iterrows():
#     k = item['spu_sn']
#     cnt += 1
#     v1 = src_redis.get(k)
#     if v1 is not None:
#         target_redis.set(k, v1)
#     if cnt % 1000 == 0:
#         print(cnt, flush=True)

key_list = src_redis.keys()

with src_redis.pipeline() as pipe_src:
    num = 1000  # 定义每组包含的元素个数
    j = 1
    for i in range(0, len(key_list), num):
        curr_key_list = key_list[i:i + num]
        for key in curr_key_list:
            pipe_src.get(key)
        curr_result = pipe_src.execute()
        with target_redis.pipeline() as pipe_tgt:
            for k, v in zip(curr_key_list, curr_result):
                if v is not None and k is not None:
                    pipe_tgt.set(k,v)
                    pipe_tgt.expire(k, 7*24*3600)
            pipe_tgt.execute()
        print(j*num, flush=True)
        j+=1

