import keras
import pandas as pd
from keras import Model as mod
from keras.utils.np_utils import to_categorical
from keras.layers import Input, LSTM, Dense, merge, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import numpy as np
from keras.preprocessing.text import Tokenizer
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 200000
VALIDATION_SPLIT = 0.5
EMBEDDING_DIM = 200
spa = pd.read_csv('/home/moon/work/tianchi/data/train.txt', sep = '\t', header = None)
spa.columns = ['spa_qura1', 'eng_qura1', 'spa_qura2', 'eng_qura2', 'label']
data=spa[['spa_qura1','spa_qura2','label']]
# for row in data.rows:
#     print(row['spa_qura1'])
#data['spa_qura_list_1'] = data['spa_qura1'].apply(lambda x : x.split(' '))
labels =[]
train_left = []
train_right = []
embeddings_index = {}
texts = []
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
def _map(data):
    for index, row in data.iterrows():   # 获取每行的index、row
        texts.append(row['spa_qura1'])
        texts.append(row['spa_qura2'])
        labels.append(row['label'])
        train_left.append(row['spa_qura1'])
        train_right.append(row['spa_qura2'])
_map(data)

tokenizer.fit_on_texts(texts)
#print(data['spa_qura1'])
sequences_left = tokenizer.texts_to_sequences(train_left)
sequences_right = tokenizer.texts_to_sequences(train_right)


word_index = tokenizer.word_index
data_left = pad_sequences(sequences_left, maxlen=MAX_SEQUENCE_LENGTH,padding='pre', truncating='post')
data_right = pad_sequences(sequences_right, maxlen=MAX_SEQUENCE_LENGTH,padding='pre', truncating='post')
labels = np.array(labels)
indices = np.arange(data_left.shape[0])
np.random.shuffle(indices)
data_left = data_left[indices]
data_right = data_right[indices]
labels = labels[indices]
#labels = labels[0]
nb_validation_samples = int(VALIDATION_SPLIT * data_left.shape[0])
input_train_left = data_left[:-nb_validation_samples]
input_train_right = data_right[:-nb_validation_samples]

val_left = data_left[-nb_validation_samples:]
val_right = data_right[-nb_validation_samples:]

labels_train = labels[-nb_validation_samples:]
labels_val = labels[-nb_validation_samples:]
nb_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector #


embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            trainable=True)

tweet_a = Input(shape=(MAX_SEQUENCE_LENGTH,))
tweet_b = Input(shape=(MAX_SEQUENCE_LENGTH,))
tweet_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
#下面这些行是神经网络构造的内容，可参见上面的网络设计图
embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embedding_matrix], trainable=True)(tweet_input)

#看不懂
conv1 = Conv1D(128, 3, activation='tanh')(embedding_layer)
drop_1 = Dropout(0.2)(conv1)
max_1 = MaxPooling1D(3)(drop_1)
conv2 = Conv1D(128, 3, activation='tanh')(max_1)
drop_2 = Dropout(0.2)(conv2)
max_2 = MaxPooling1D(3)(drop_2)
#conv2 = Conv1D(128, 3, activation='tanh')(max_1)
#max_2 = MaxPooling1D(3)(conv2)
out_1 = Flatten()(max_1)
#out_1 = LSTM(128)(max_1)
model_encode = mod(tweet_input, out_1) # 500(examples) * 5888
encoded_a = model_encode(tweet_a)
encoded_b = model_encode(tweet_b)
merged_vector = keras.layers.Add()([encoded_a, encoded_b]) # good
dense_1 = Dense(128,activation='relu')(merged_vector)
dense_2 = Dense(128,activation='relu')(dense_1)
dense_3 = Dense(128,activation='relu')(dense_2)
predictions = Dense(1, activation='sigmoid')(dense_3)


model = mod(input=[tweet_a, tweet_b], output=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 下面是训练程序

model.fit([input_train_left,input_train_right], labels_train, nb_epoch=5)
#print(model.fit([input_train_left,input_train_right], labels_train, nb_epoch=5))
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')
# 下面是训练得到的神经网络进行评估
score = model.evaluate([input_train_left,input_train_right], labels_train, verbose=0)
print('train score:', score[0]) # 训练集中的loss
print('train accuracy:', score[1]) # 训练集中的准确率
score = model.evaluate([val_left, val_right], labels_val, verbose=0)
print('Test score:', score[0])#测试集中的loss
print('Test accuracy:', score[1]) #测试集中的准确率
testdata=[]
testdata.append("aaaaa")
testcompare=[]
testcompare.append("bbbbb")
a = model.predict([testdata,testcompare])
print(a)