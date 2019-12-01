import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import gc
import os
import time
import multiprocessing as mp

from sklearn.preprocessing import LabelEncoder

tic = time.time()

SAVE_PATH = './feats'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)


##################  member_info: 用户特征¶

with open('../pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)
member_info.head(2)

# 原始类别特征
member_cat_feats = ['gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2']
for feat in member_cat_feats:
    member_info[feat] = LabelEncoder().fit_transform(member_info[feat])

# 用户关注和感兴趣的topic数
member_info['num_atten_topic'] = member_info['topic_attent'].apply(len)
member_info['num_interest_topic'] = member_info['topic_interest'].apply(len)


def most_interest_topic(d):
    if len(d) == 0:
        return -1
    return list(d.keys())[np.argmax(list(d.values()))]

# 用户最感兴趣的topic
member_info['most_interest_topic'] = member_info['topic_interest'].apply(most_interest_topic)
member_info['most_interest_topic'] = LabelEncoder().fit_transform(member_info['most_interest_topic'])


def get_interest_values(d):
    if len(d) == 0:
        return [0]
    return list(d.values())


# 用户topic兴趣值的统计特征
member_info['interest_values'] = member_info['topic_interest'].apply(get_interest_values)
member_info['min_interest_values'] = member_info['interest_values'].apply(np.min)
member_info['max_interest_values'] = member_info['interest_values'].apply(np.max)
member_info['mean_interest_values'] = member_info['interest_values'].apply(np.mean)
member_info['std_interest_values'] = member_info['interest_values'].apply(np.std)

# 汇总
feats = ['author_id', 'gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2', 'score']
feats += ['num_atten_topic', 'num_interest_topic', 'most_interest_topic']
feats += ['min_interest_values', 'max_interest_values', 'mean_interest_values', 'std_interest_values']
member_feat = member_info[feats]

member_feat.head(3)

member_feat.to_hdf('./feats/member_feat.h5', key='data')

del member_feat, member_info
gc.collect()


##################    question_info: 问题特征

with open('../pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

question_info.head(2)


# title、desc词计数，topic计数
question_info['num_title_sw'] = question_info['title_sw_series'].apply(len)
question_info['num_title_w'] = question_info['title_w_series'].apply(len)
question_info['num_desc_sw'] = question_info['desc_sw_series'].apply(len)
question_info['num_desc_w'] = question_info['desc_w_series'].apply(len)
question_info['num_qtopic'] = question_info['topic'].apply(len)

feats = ['question_id', 'num_title_sw', 'num_title_w', 'num_desc_sw', 'num_desc_w', 'num_qtopic', 'question_hour']
feats += []
question_feat = question_info[feats]

question_feat.head(3)

question_feat.to_hdf('./feats/question_feat.h5', key='data')

del question_info, question_feat
gc.collect()

################# member_info & question_info: 用户和问题的交互特征

with open('../pkl/invite_info.pkl', 'rb') as file:
    invite_info = pickle.load(file)
with open('../pkl/invite_info_evaluate.pkl', 'rb') as file:
    invite_info_evaluate = pickle.load(file)
with open('../pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)
with open('../pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)


# 合并 author_id，question_id
invite = pd.concat([invite_info, invite_info_evaluate])
invite_id = invite[['author_id', 'question_id']]
invite_id['author_question_id'] = invite_id['author_id'] + invite_id['question_id']
invite_id.drop_duplicates(subset='author_question_id',inplace=True)
invite_id_qm = invite_id.merge(member_info[['author_id', 'topic_attent', 'topic_interest']], 'left', 'author_id').merge(question_info[['question_id', 'topic']], 'left', 'question_id')
invite_id_qm.head(2)


# 分割 df，方便多进程跑
def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

def gc_mp(pool, ret, chunk_list):
    del pool
    for r in ret:
        del r
    del ret
    for cl in chunk_list:
        del cl
    del chunk_list
    gc.collect()

# 用户关注topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_attent']) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_attent_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)


# 用户感兴趣topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_interest'].keys()) & set(row['topic'])),axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# 用户感兴趣topic和问题 topic的交集的兴趣值
def process(df):
    return df.apply(lambda row: [row['topic_interest'][t] for t in row['topic_interest_intersection']],axis=1)

pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)


# 交集topic计数
invite_id_qm['num_topic_attent_intersection'] = invite_id_qm['topic_attent_intersection'].apply(len)
invite_id_qm['num_topic_interest_intersection'] = invite_id_qm['topic_interest_intersection'].apply(len)


# 交集topic兴趣值统计
invite_id_qm['topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(lambda x: [0] if len(x) == 0 else x)
invite_id_qm['min_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.min)
invite_id_qm['max_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.max)
invite_id_qm['mean_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.mean)
invite_id_qm['std_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(np.std)


feats = ['author_question_id', 'num_topic_attent_intersection', 'num_topic_interest_intersection', 'min_topic_interest_intersection_values', 'max_topic_interest_intersection_values', 'mean_topic_interest_intersection_values', 'std_topic_interest_intersection_values']
feats += []
member_question_feat = invite_id_qm[feats]
member_question_feat.head(3)

member_question_feat.to_hdf('./feats/member_question_feat.h5', key='data')

del invite_id_qm, member_question_feat
gc.collect()

toc = time.time()
print('Used time: %d' % int(toc-tic))


##################  

