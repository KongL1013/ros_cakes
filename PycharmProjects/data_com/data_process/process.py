import pandas as pd
import numpy as np
import pickle
import gc
from tqdm import tqdm_notebook
import os
import time

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)





tic = time.time()
# 减少内存占用
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# 解析列表， 重编码id
def parse_str(d):
    return np.array(list(map(float, d.split())))

def parse_list_1(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[1:]), str(d).split(',')))

def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))

def parse_map(d):
    if d == '-1':
        return {}
    return dict([int(z.split(':')[0][1:]), float(z.split(':')[1])] for z in d.split(','))

PATH = '../kanshanbei'
SAVE_PATH = '../pkl'
if not os.path.exists(SAVE_PATH):
    print('create dir: %s' % SAVE_PATH)
    os.mkdir(SAVE_PATH)

#########################   single_word
single_word = pd.read_csv(os.path.join(PATH, 'single_word_vectors_64d.txt'),
                          names=['id', 'embed'], sep='\t')
print(single_word.head())
single_word['embed'] = single_word['embed'].apply(parse_str)
single_word['id'] = single_word['id'].apply(lambda x: int(x[2:]))
print(single_word.head())
print('single_word')
with open('../pkl/single_word.pkl', 'wb') as file:
    pickle.dump(single_word, file)

del single_word
gc.collect() #gc 垃圾回收模块



#########################   word
word = pd.read_csv(os.path.join(PATH,'word_vectors_64d.txt'),names=['id','embed'],sep='\t')
print('word')
print(word.head())
print(word['embed'][1])
print(type(word['embed'][1]))
word['embed'] = word['embed'].apply(parse_str) #把str转为float
with open(os.path.join(SAVE_PATH,'word.pkl'),'wb') as file:
    pickle.dump(word,file)
del word
gc.collect()


########################   topic
topic = pd.read_csv(os.path.join(PATH, 'topic_vectors_64d.txt'),
                          names=['id', 'embed'], sep='\t')
print('topic')
print(topic.head())

topic['embed'] = topic['embed'].apply(parse_str)
topic['id'] = topic['id'].apply(lambda x: int(x[1:]))
topic.head()

with open('../pkl/topic.pkl', 'wb') as file:
    pickle.dump(topic, file)

del topic
gc.collect()

########################   invite


invite_info = pd.read_csv(os.path.join(PATH, 'invite_info_0926.txt'),
                          names=['question_id', 'author_id', 'invite_time', 'label'], sep='\t')
invite_info_evaluate = pd.read_csv(os.path.join(PATH, 'invite_info_evaluate_1_0926.txt'),
                          names=['question_id', 'author_id', 'invite_time'], sep='\t')
print('invite_info')
print(invite_info.head())
invite_info['day'] = invite_info['invite_time']\
    .apply(lambda x: int(x.split('-')[0][1:])).astype(np.int8)
invite_info['invite_hour'] = invite_info['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

invite_info_evaluate['invite_day'] = invite_info_evaluate['invite_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
invite_info_evaluate['invite_hour'] = invite_info_evaluate['invite_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)

invite_info = reduce_mem_usage(invite_info)

with open('../pkl/invite_info.pkl', 'wb') as file:
    pickle.dump(invite_info, file)

with open('../pkl/invite_info_evaluate.pkl', 'wb') as file:
    pickle.dump(invite_info_evaluate, file)

del invite_info, invite_info_evaluate
gc.collect()


################################   member

member_info = pd.read_csv(os.path.join(PATH, 'member_info_0926.txt'),
                          names=['author_id', 'gender', 'keyword', 'grade', 'hotness', 'reg_type','reg_plat','freq',
                                 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2',
                                 'score', 'topic_attent', 'topic_interest'], sep='\t')
print('member_info')
print(member_info.head())


member_info['topic_attent'] = member_info['topic_attent'].apply(parse_list_1)
member_info['topic_interest'] = member_info['topic_interest'].apply(parse_map)

member_info = reduce_mem_usage(member_info)

with open('../pkl/member_info.pkl', 'wb') as file:
    pickle.dump(member_info, file)

del member_info
gc.collect()


##################### question_info

question_info = pd.read_csv(os.path.join(PATH, 'question_info_0926.txt'),
                          names=['question_id', 'question_time', 'title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series', 'topic'], sep='\t')
print(question_info.head())

question_info['title_sw_series'] = question_info['title_sw_series'].apply(parse_list_2)#.apply(sw_lbl_enc.transform).apply(list)
question_info['title_w_series'] = question_info['title_w_series'].apply(parse_list_1)#.apply(w_lbl_enc.transform).apply(list)
question_info['desc_sw_series'] = question_info['desc_sw_series'].apply(parse_list_2)#.apply(sw_lbl_enc.transform).apply(list)
question_info['desc_w_series'] = question_info['desc_w_series'].apply(parse_list_1)#.apply(w_lbl_enc.transform).apply(list)
question_info['topic'] = question_info['topic'].apply(parse_list_1)# .apply(topic_lbl_enc.transform).apply(list)

question_info['question_day'] = question_info['question_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
question_info['question_hour'] = question_info['question_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
del question_info['question_time']
gc.collect()

print(question_info.head())

question_info = reduce_mem_usage(question_info)

with open('../pkl/question_info.pkl', 'wb') as file:
    pickle.dump(question_info, file)

del question_info
gc.collect()


################################  answer
#%%time
print('answer')
answer_info = pd.read_csv(os.path.join(PATH, 'answer_info_0926.txt'),
                          names=['answer_id', 'question_id', 'author_id', 'answer_time', 'content_sw_series', 'content_w_series',
                                 'excellent', 'recommend', 'round_table', 'figure', 'video',
                                 'num_word', 'num_like', 'num_unlike', 'num_comment',
                                 'num_favor', 'num_thank', 'num_report', 'num_nohelp', 'num_oppose'], sep='\t')
answer_info.head()

answer_info['content_sw_series'] = answer_info['content_sw_series'].apply(parse_list_2)
answer_info['content_w_series'] = answer_info['content_w_series'].apply(parse_list_1)
answer_info.head()


answer_info['answer_day'] = answer_info['answer_time'].apply(lambda x: int(x.split('-')[0][1:])).astype(np.int16)
answer_info['answer_hour'] = answer_info['answer_time'].apply(lambda x: int(x.split('-')[1][1:])).astype(np.int8)
del answer_info['answer_time']
gc.collect()


answer_info = reduce_mem_usage(answer_info)



with open('../pkl/answer_info.pkl', 'wb') as file:
    pickle.dump(answer_info, file)

del answer_info
gc.collect()
toc = time.time()
print('Used time: %d' % int(toc-tic))










