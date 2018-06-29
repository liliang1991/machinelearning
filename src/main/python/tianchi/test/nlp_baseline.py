from gensim.models import Word2Vec
import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import log_loss
import math
import networkx as nx
def lgb_model(X,y,test):
	N = 2
	skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=5)

	xx_cv = []
	xx_pre = []

    # specify your configurations as a dict
	params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 60,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
   	}

	for train_in,test_in in skf.split(X,y):
		X_train,X_test,y_train,y_test  = X[train_in],X[test_in],y[train_in],y[test_in]
		lgb_train = lgb.Dataset(X_train, y_train)
		lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

		print('Start training...')
    	# train
		gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

		print('Start predicting...')
		y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
		xx_cv.append(log_loss(y_test,y_pred))
		xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))
	return xx_cv, xx_pre

english_spa = pd.read_csv('/home/moon/work/tianchi/data/cikm_english_train_20180516.txt', sep = '\t', header = None)
spa = pd.read_csv('/home/moon/work/tianchi/data/cikm_spanish_train_20180516.txt', sep = '\t', header = None)
test = pd.read_csv('/home/moon/work/tianchi/data/cikm_test_a_20180516.txt', sep = '\t', header = None)
test.columns = ['spa_qura1', 'spa_qura2']
english_spa.columns = ['eng_qura1', 'spa_qura1', 'eng_qura2', 'spa_qura2', 'label']
spa.columns = ['spa_qura1', 'eng_qura1', 'spa_qura2', 'eng_qura2', 'label']
test['label'] = -1

english = english_spa[['eng_qura1','eng_qura2','label']]
english.columns = ['spa_qura1','spa_qura2','label']

english1 = spa[['eng_qura1','eng_qura2','label']]
english1.columns = ['spa_qura1','spa_qura2','label']
pd.set_option('display.max_rows',None)
data = pd.concat([test, english_spa[test.columns], spa[test.columns],english[english.label == 1][test.columns]]).reset_index()


data['spa_qura_list_1'] = data['spa_qura1'].apply(lambda x : x.split(' '))
data['spa_qura_list_2'] = data['spa_qura2'].apply(lambda x : x.split(' '))

spa_list = list(data['spa_qura_list_1'])
spa_list.extend(list(data['spa_qura_list_2']))

model = Word2Vec(spa_list, sg=1, size=30,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)

def seq_to_w2v(seq, model):
	words = []
	default = [0 for x in range(30)]
	for i in range(30):
		if i < len(seq):
			words.extend(model[seq[i]])
		else:
			words.extend(default)
	return words

data['spa_list_1'] = data['spa_qura_list_1'].apply(lambda x : seq_to_w2v(x, model))
data['spa_list_2'] = data['spa_qura_list_2'].apply(lambda x : seq_to_w2v(x, model))
data[data.label == -1].to_csv('test.csv',index = None)
data[data.label == 1].to_csv('train.csv', index = None)


def calEuclideanDistance(vec1,vec2):  
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))  
    return dist 

def Manhattan(vec1, vec2):# Manhattan_Distance,曼哈顿距离
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return np.abs(npvec1-npvec2).sum()

def Cosine(vec1, vec2):
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    return npvec1.dot(npvec2)/(math.sqrt((npvec1**2).sum()) * math.sqrt((npvec2**2).sum()))

def w2v_similar(data):
#	print(np.array(data['spa_qura_w2v_1']))
	#print(calEuclideanDistance(np.array(data['spa_qura_w2v_1']), np.array(data['spa_qura_w2v_2'])))
	return calEuclideanDistance(np.array(data['spa_list_1']), np.array(data['spa_list_2']))

def w2v_manha_similar(data):
	return Manhattan(data['spa_list_1'], data['spa_list_2'])

def w2v_cos_similar(data):
	return Cosine(data['spa_list_1'], data['spa_list_2'])

def str_similar(data):
	return Edit_distance_str(data['spa_qura1'], data['spa_qura2'])

def Edit_distance_str(str1, str2):
   import Levenshtein
   edit_distance_distance = Levenshtein.distance(str1, str2)
   similarity = 1-(edit_distance_distance/max(len(str1), len(str2)))
   return {'Distance': edit_distance_distance, 'Similarity': similarity}
#print(data.apply(w2v_similar, axis = 1))
#axis=0表述列
#axis=1表述行
#print(np.concatenate([w2v_similar],axis=1))
print(data.iloc[0])

data['spa_w2v_similar'] = data.apply(w2v_similar, axis = 1)
data['spa_w2v_manha_similar'] = data.apply(w2v_manha_similar, axis = 1)
data['spa_w2v_cos_similar'] = data.apply(w2v_cos_similar, axis = 1)
#data['spa_str_edit_similar'] = data.apply(str_similar, axis = 1)

def count_cha(data):
	return len(data['spa_qura_list_1']) - len(data['spa_qura_list_1'])

data['count1'] = data['spa_qura_list_1'].apply(lambda x : len(x))
data['count2'] = data['spa_qura_list_2'].apply(lambda x : len(x))

data['count_cha'] = data.apply(count_cha, axis = 1)

data['seq_len_1'] =  data['spa_qura1'].apply(lambda x : len(x.split(',')))
data['seq_len_2'] =  data['spa_qura2'].apply(lambda x : len(x.split(',')))

data['seq_len_cha'] = data['seq_len_1'] - data['seq_len_2']

data['jiaoji_cnt'] = data.apply(lambda x : len(list(set(x['spa_qura1']).intersection(set(x['spa_qura2'])))), axis = 1)
data['jiaoji_cnt_rate1'] = data.apply(lambda x : float(x['jiaoji_cnt']) / float(len(x['spa_qura1'])),axis = 1)
data['jiaoji_cnt_rate2'] = data.apply(lambda x : float(x['jiaoji_cnt']) / float(len(x['spa_qura2'])),axis = 1)
data['jiaoji_cnt_rate_char'] = data['jiaoji_cnt_rate1'] - data['jiaoji_cnt_rate2']

feature = ['spa_w2v_cos_similar','spa_w2v_manha_similar','spa_w2v_similar', 'count1', 'count2', 'count_cha', 'seq_len_1', 'seq_len_2', 'seq_len_cha','jiaoji_cnt','jiaoji_cnt_rate1','jiaoji_cnt_rate2','jiaoji_cnt_rate_char']


train = data[data['label']!=-1]
test = data[data['label']==-1]

y = train.pop('label').values
X = train[feature].values

test_y = test.pop('label').values
test1 = test[feature].values

result, tt = lgb_model(X, y, test1)
print(len(tt))
s = 0
for i in tt:
        #print (i)
	s = s + i
s = s /5

test['label'] = list(s)
print (result)
print (np.mean(result))
print (len(test))

test[['label']].to_csv('/home/moon/work/tianchi/data/result.txt',index = None, header = None)
