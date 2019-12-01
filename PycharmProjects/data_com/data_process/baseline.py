import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import gc
import pickle
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from catboost import CatBoostClassifier, Pool

tic = time.time()

with open('../pkl/invite_info.pkl', 'rb') as file:
    invite_info = pickle.load(file)
with open('../pkl/invite_info_evaluate.pkl', 'rb') as file:
    invite_info_evaluate = pickle.load(file)


member_feat = pd.read_hdf('./feats/member_feat.h5', key='data')  # 0.689438
question_feat = pd.read_hdf('./feats/question_feat.h5', key='data')  # 0.706848


member_question_feat = pd.read_hdf('./feats/member_question_feat.h5', key='data')  # 719116 d12
invite_info['author_question_id'] = invite_info['author_id'] + invite_info['question_id']
invite_info_evaluate['author_question_id'] = invite_info_evaluate['author_id'] + invite_info_evaluate['question_id']


train = invite_info.merge(member_feat, 'left', 'author_id')
test = invite_info_evaluate.merge(member_feat, 'left', 'author_id')

train = train.merge(question_feat, 'left', 'question_id')
test = test.merge(question_feat, 'left', 'question_id')

train = train.merge(member_question_feat, 'left', 'author_question_id')
test = test.merge(member_question_feat, 'left', 'author_question_id')

del member_feat, question_feat, member_question_feat
gc.collect()

drop_feats = ['question_id', 'author_id', 'author_question_id', 'invite_time', 'label', 'invite_day']

used_feats = [f for f in train.columns if f not in drop_feats]
print(len(used_feats))
print(used_feats)

train_x = train[used_feats].reset_index(drop=True)
train_y = train['label'].reset_index(drop=True)
test_x = test[used_feats].reset_index(drop=True)

preds = np.zeros((test_x.shape[0], 2))
scores = []
has_saved = False
imp = pd.DataFrame()
imp['feat'] = used_feats

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for index, (tr_idx, va_idx) in enumerate(kfold.split(train_x, train_y)):
    print('*' * 30)
    X_train, y_train, X_valid, y_valid = train_x.iloc[tr_idx], train_y.iloc[tr_idx], train_x.iloc[va_idx], train_y.iloc[
        va_idx]
    cate_features = []
    train_pool = Pool(X_train, y_train, cat_features=cate_features)
    eval_pool = Pool(X_valid, y_valid, cat_features=cate_features)
    if not has_saved:
        cbt_model = CatBoostClassifier(iterations=10000,
                                       learning_rate=0.1,
                                       eval_metric='AUC',
                                       use_best_model=True,
                                       random_seed=42,
                                       logging_level='Verbose',
                                       task_type='GPU',
                                       devices='0',
                                       early_stopping_rounds=300,
                                       loss_function='Logloss',
                                       depth=12,
                                       )
        cbt_model.fit(train_pool, eval_set=eval_pool, verbose=100)
    #         with open('./models/fold%d_cbt_v1.mdl' % index, 'wb') as file:
    #             pickle.dump(cbt_model, file)
    else:
        with open('./models/fold%d_cbt_v1.mdl' % index, 'rb') as file:
            cbt_model = pickle.load(file)

    imp['score%d' % (index + 1)] = cbt_model.feature_importances_

    score = cbt_model.best_score_['validation']['AUC']
    scores.append(score)
    print('fold %d round %d : score: %.6f | mean score %.6f' % (
    index + 1, cbt_model.best_iteration_, score, np.mean(scores)))
    preds += cbt_model.predict_proba(test_x)

    del cbt_model, train_pool, eval_pool
    del X_train, y_train, X_valid, y_valid
    import gc

    gc.collect()

#     mdls.append(cbt_model)

imp.sort_values(by='score1', ascending=False)

result = invite_info_evaluate[['question_id', 'author_id', 'invite_time']]
result['result'] = preds[:, 1] / 5
result.head()

result.to_csv('./result.txt', sep='\t', index=False, header=False)

toc = time.time()
print('Used time: %d' % int(toc - tic))