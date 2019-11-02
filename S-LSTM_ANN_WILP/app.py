# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 12:04:49 2019

@author: Skanda
"""

import numpy as np
import pandas as pd
from time import time
from gensim.models import Word2Vec 
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import itertools
from operator import itemgetter
import datetime
import gc
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta

import pandas as pd
import heapq
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.models import load_model

TRAIN_CSV = 'Dataset WILP/dataset.csv'
MODEL_SAVING_DIR = './'

def get_word2vec(documents, embedding_dim):
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors

df = pd.read_csv(TRAIN_CSV)
questions = list(df['Questions'])
answers = list(df['Answers'])
similarity = list(df['similarity'])
X = df.iloc[:, 1:3]
y = df.iloc[:, 4]
del df
documents = questions + answers
tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as fp:
    tokenizer = pickle.load(fp)
embedding_dim = 100
word_vector = None
with open('word2vec.pickle', 'rb') as fp:
    word_vector = pickle.load(fp)
embeddings = np.random.randn(len(tokenizer.word_index) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored
vocabulary = tokenizer.word_index
for word, i in vocabulary.items():
    word = word.lower()
    if word in word_vector.vocab:
        embeddings[i] = word_vector[word]
print('Null word embeddings: %d' % np.sum(np.sum(embeddings, axis=1) == 0))    
del word_vector
with open('embeddings.pickle', 'wb') as fp:
    pickle.dump(embeddings, fp)
gc.collect()
            
max_seq_length = max(max([len(x.split(' ')) for x in questions]), max([len(x.split(' ')) for x in answers]))
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2)

train_left = tokenizer.texts_to_sequences(list(X_train['Questions']))
train_right = tokenizer.texts_to_sequences(list(X_train['Answers']))
validation_left = tokenizer.texts_to_sequences(list(X_validation['Questions']))
validation_right = tokenizer.texts_to_sequences(list(X_validation['Answers']))
X_train = {'left': train_left, 'right': train_right}
X_validation = {'left': validation_left, 'right': validation_right}

# Convert labels to their numpy representations
y_train = y_train.values
y_validation = y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(y_train)

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 15


# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = Bidirectional(LSTM(n_hidden, dropout=0.17, recurrent_dropout=0.17))

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

merged = concatenate([left_output, right_output])
merged = BatchNormalization()(merged)
merged = Dropout(0.20)(merged)
merged = Dense(20, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.20)(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[left_input, right_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)


model_trained = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=n_epoch,
                            shuffle=True, validation_data=([X_validation['left'], X_validation['right']], y_validation))


model.save('SLSTM.h5')




def questions_vs_answers(test_question, category):
    max_seq_length = 125
    with open('tokenizer.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    model = load_model('SBiLSTM-ANN.h5')
    data = pd.read_csv('Dataset WILP/dataset.csv')
    data = data.loc[(data['category'] == category) & (data['similarity'] == 1)]
    test_questions = []
    dataset_questions = []
    test_answers = []
    test_indices = []
    for index,row in data.iterrows():
        test_questions.append(test_question)
        test_answers.append(row['Answers'])
        dataset_questions.append(row['Questions'])
        test_indices.append(index)
    test_data_left = pad_sequences(tokenizer.texts_to_sequences(test_questions), maxlen = max_seq_length)
    test_data_right = pad_sequences(tokenizer.texts_to_sequences(test_answers), maxlen = max_seq_length)
    preds = list(model.predict([test_data_left, test_data_right], verbose=1).ravel())
    indexed_preds = list(zip(test_indices, preds))
    indexed_preds.sort(key=itemgetter(1), reverse=True)
    result_questions = []
    result_answers= []
    for tup in indexed_preds:
        index = tup[0]
        result_questions.append(dataset_questions[test_indices.index(index)])
        result_answers.append(test_answers[test_indices.index(index)])
    results = [(x, y, w[0], w[1]) for (x, y), w in zip(zip(result_questions, result_answers), indexed_preds)]
    with open('RESULTSQA.txt', 'w+') as fp:
        i = 1
        fp.write('Question is ')
        fp.write(test_question)
        fp.write('\n')
        fp.write('++++++++++++++++++++++++++++++++\n')
        for quad in results:
            fp.write('Result ' + str(i))
            fp.write('\n')
            i = i + 1
            fp.write(quad[0])
            fp.write('\n')
            fp.write(quad[1])
            fp.write('\nSimilarity Score : ' + str(quad[3]))
            fp.write('\n')
            fp.write('++++++++++++++++++++++++++++++++\n')
            

def questions_vs_questions(test_question, category):
    max_seq_length = 125
    with open('tokenizer.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    model = load_model('SBiLSTM-ANN.h5')
    data = pd.read_csv('Dataset WILP/dataset.csv')
    data = data.loc[(data['category'] == category) & (data['similarity'] == 1)]
    test_questions = []
    dataset_questions = []
    test_answers = []
    test_indices = []
    for index,row in data.iterrows():
        test_questions.append(test_question)
        test_answers.append(row['Answers'])
        dataset_questions.append(row['Questions'])
        test_indices.append(index)
    test_data_left = pad_sequences(tokenizer.texts_to_sequences(test_questions), maxlen = max_seq_length)
    test_data_right = pad_sequences(tokenizer.texts_to_sequences(dataset_questions), maxlen = max_seq_length)
    preds = list(model.predict([test_data_left, test_data_right], verbose=1).ravel())
    indexed_preds = list(zip(test_indices, preds))
    indexed_preds.sort(key=itemgetter(1), reverse=True)
    result_questions = []
    result_answers= []
    for tup in indexed_preds:
        index = tup[0]
        result_questions.append(dataset_questions[test_indices.index(index)])
        result_answers.append(test_answers[test_indices.index(index)])
    results = [(x, y, w[0], w[1]) for (x, y), w in zip(zip(result_questions, result_answers), indexed_preds)]
    with open('RESULTSQQ.txt', 'w+') as fp:
        i = 1
        fp.write('Question is ')
        fp.write(test_question)
        fp.write('\n')
        fp.write('++++++++++++++++++++++++++++++++\n')
        for quad in results:
            fp.write('Result ' + str(i))
            fp.write('\n')
            i = i + 1
            fp.write(quad[0])
            fp.write('\n')
            fp.write(quad[1])
            fp.write('\nSimilarity Score : ' + str(quad[3]))
            fp.write('\n')
            fp.write('++++++++++++++++++++++++++++++++\n')
            

def main():
   test_question = 'the engine is not running'
   category = 'J-octa'
   questions_vs_answers(test_question, category)
   questions_vs_questions(test_question, category)
   

if __name__  == '__main__':
    main()