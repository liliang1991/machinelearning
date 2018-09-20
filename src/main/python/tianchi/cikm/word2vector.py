import logging
from gensim.models.word2vec import LineSentence, Word2Vec
import sys
import pandas as pd
#加载数据
# 英语问句对  英语问句1，西班牙语翻译1，英语问句2，西班牙语翻译2，匹配标注。
english_spa = pd.read_csv('/home/moon/work/tianchi/data/cikm_english_train_20180516.txt', sep = '\t', header = None)
english_spa.columns = ['eng_qura1', 'spa_qura1', 'eng_qura2', 'spa_qura2', 'label']
#西班牙语问句1
english_spa['spa_qura_list_1'] = english_spa['spa_qura1'].apply(lambda x : x.split(' '))
#西班牙语问句2
english_spa['spa_qura_list_2'] = english_spa['spa_qura2'].apply(lambda x : x.split(' '))
spa_list = list(english_spa['spa_qura_list_1'])
spa_list.extend(list(english_spa['spa_qura_list_2']))
model = Word2Vec(spa_list, sg=1, size=30,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=8)
model.save("./w2v.mod")