#https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.6acd33afsWeJrz&raceId=231661
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from gensim.models import Word2Vec
import codecs
import csv
import numpy as np
from keras.models import Model
#声明变量
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 30
#加载数据
# 英语问句对  英语问句1，西班牙语翻译1，英语问句2，西班牙语翻译2，匹配标注。
english_spa = pd.read_csv('/home/moon/work/tianchi/data/cikm_english_train_head.txt', sep = '\t', header = None)
#西班牙问句对 西班牙语问句1，英语翻译1，西班牙语问句2，英语翻译2，匹配标注
spa = pd.read_csv('/home/moon/work/tianchi/data/cikm_test_head_20180516.txt', sep = '\t', header = None)
print("Fit tokenizer...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
train_file="/home/moon/work/tianchi/data/cikm_english_train_20180516.txt"
test_file="/home/moon/work/tianchi/data/cikm_test_a_20180516.txt"
EMBEDDING_FILE = '/home/moon/work/workspace/ky/tensorlfow/src/main/python/tianchi/cikm/w2v.mod'
texts_1 = []
texts_2 = []
labels = []

with codecs.open(train_file, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for values in reader:
        texts_1.append(values[1])
        texts_2.append(values[3])
        labels.append(int(values[4]))


test_texts_1 = []
test_texts_2=[]
test_ids = []
i=-1;

with codecs.open(test_file, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for values in reader:
        i+=1
        test_texts_1.append(values[0])
        test_texts_2.append(values[1])
        test_ids.append(i)
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
#word_index = tokenizer.word_index
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)
sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2=tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)
labels = np.array(labels)
#设置30 的长度，不足的前面补０
data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
#embedding_matrix.shape (40, 100) 40个数组，每个１００个元素
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
word2vec = Word2Vec.load(EMBEDDING_FILE)
for word, i in word_index.items():
    if word in word2vec.wv.vocab:
        embedding_matrix[i] = word2vec.wv.word_vec(word)
    else:
        print(word)
#lstm 参数
num_lstm = 175
num_dense = 100
rate_drop_lstm = 0.15
rate_drop_dense = 0.15
act = 'relu'
def get_model():
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
                  outputs=preds)
    #逻辑回归
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.summary()
    #model.save("../model.txt")
    print('模型写入完成')
    return model

STAMP = '../model/lstm/lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, \
                                              rate_drop_dense)
def train_model():
    model = get_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = '/home/moon/work/workspace/ky/tensorlfow/src/main/python/tianchi/cikm/model.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1, data_2], labels, \
                     validation_data=([data_1, data_2], labels), \
                     epochs=2, batch_size=5000, shuffle=True, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    predicts = model.predict([test_data_1, test_data_2], batch_size=10, verbose=1)
    for i in range(len(test_ids)):
        print("t1: %s, t2: %s, score: %s" % (test_texts_1[i], test_texts_2[i], predicts[i]))
    bst_score = min(hist.history['loss'])

    bst_acc = max(hist.history['acc'])
    print(bst_acc, bst_score)
if __name__ == '__main__':
    train_model()

