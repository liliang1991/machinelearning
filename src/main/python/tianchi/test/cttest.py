import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 200000
spa = pd.read_csv('/home/moon/work/tianchi/data/train.txt', sep = '\t', header = None)
spa.columns = ['spa_qura1', 'eng_qura1', 'spa_qura2', 'eng_qura2', 'label']
data=spa[['spa_qura1','spa_qura2','label']]
# for row in data.rows:
#     print(row['spa_qura1'])
#data['spa_qura_list_1'] = data['spa_qura1'].apply(lambda x : x.split(' '))
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data['spa_qura1'][0])
#print(data['spa_qura1'])
sequences_left = tokenizer.texts_to_sequences(data['spa_qura1'][0])
print(sequences_left)
data_left = pad_sequences(sequences_left, maxlen=MAX_SEQUENCE_LENGTH,padding='pre', truncating='post')
print(data_left)
