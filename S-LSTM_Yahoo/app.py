# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 02:17:44 2019

@author: Skanda
"""

import numpy as np
import pandas as pd
from time import time
from gensim.models import Word2Vec 
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

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.models import load_model

TRAIN_CSV = 'dataset.csv'
MODEL_SAVING_DIR = './'

def get_word2vec(documents, embedding_dim):
    model = Word2Vec(documents, min_count=1, size=embedding_dim)
    word_vectors = model.wv
    del model
    return word_vectors

df = pd.read_csv(TRAIN_CSV)
questions = list(df['Question'])
answers = list(df['Answer'])
similarity = list(df['Similarity'])
X = df.iloc[:, 2:4]
y = df.iloc[:, 5]
del df
documents = questions + answers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
with open('tokenizer.pickle', 'wb') as fp:
    pickle.dump(tokenizer, fp)
embedding_dim = 100
word_vector = get_word2vec(documents, embedding_dim)
embeddings = np.random.randn(len(tokenizer.word_index) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored
vocabulary = tokenizer.word_index
for word, i in vocabulary.items():
    word = word.lower()
    if word in word_vector.vocab:
        embeddings[i] = word_vector[word]
print('Null word embeddings: %d' % np.sum(np.sum(embeddings, axis=1) == 0))    
del word_vector
gc.collect()
            
max_seq_length = max(max([len(x.split(' ')) for x in questions]), max([len(x.split(' ')) for x in answers]))
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2)

train_left = tokenizer.texts_to_sequences(list(X_train['Question']))
train_right = tokenizer.texts_to_sequences(list(X_train['Answer']))
validation_left = tokenizer.texts_to_sequences(list(X_validation['Question']))
validation_right = tokenizer.texts_to_sequences(list(X_validation['Answer']))
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
n_hidden = 20
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 10

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=n_epoch,
                            shuffle=True, validation_data=([X_validation['left'], X_validation['right']], y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


malstm.save('SLSTM.h5')




def questions_vs_answers(test_question):
    max_seq_length = 1808
    with open('tokenizer.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    custom_objects = {'exponent_neg_manhattan_distance' : exponent_neg_manhattan_distance}
    malstm = load_model('SLSTM.h5', custom_objects = custom_objects)
    data = pd.read_csv('dataset.csv')
    data = data.loc[(data['Similarity'] == 1)]
    test_questions = []
    dataset_questions = []
    test_answers = []
    test_indices = []
    for index,row in data.iterrows():
        test_questions.append(test_question)
        test_answers.append(row['Answer'])
        dataset_questions.append(row['Question'])
        test_indices.append(index)
    test_data_left = pad_sequences(tokenizer.texts_to_sequences(test_questions), maxlen = max_seq_length)
    test_data_right = pad_sequences(tokenizer.texts_to_sequences(test_answers), maxlen = max_seq_length)
    preds = list(malstm.predict([test_data_left, test_data_right], verbose=1).ravel())
    indexed_preds = list(zip(test_indices, preds))
    indexed_preds.sort(key=itemgetter(1), reverse=True)
    result_questions = []
    result_answers= []
    for tup in indexed_preds:
        index = tup[0]
        result_questions.append(dataset_questions[test_indices.index(index)])
        result_answers.append(test_answers[test_indices.index(index)])
    results = [(x, y, w[0], w[1]) for (x, y), w in zip(zip(result_questions, result_answers), indexed_preds)]
    with open('RESULTSQA.txt', 'w+', encoding='utf-8') as fp:
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
            

def questions_vs_questions(test_question):
    max_seq_length = 1808
    with open('tokenizer.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    custom_objects = {'exponent_neg_manhattan_distance' : exponent_neg_manhattan_distance}
    malstm = load_model('SLSTM.h5', custom_objects = custom_objects)
    data = pd.read_csv('dataset.csv')
    data = data.loc[(data['Similarity'] == 1)]
    test_questions = []
    dataset_questions = []
    test_answers = []
    test_indices = []
    for index,row in data.iterrows():
        test_questions.append(test_question)
        test_answers.append(row['Answer'])
        dataset_questions.append(row['Question'])
        test_indices.append(index)
    test_data_left = pad_sequences(tokenizer.texts_to_sequences(test_questions), maxlen = max_seq_length)
    test_data_right = pad_sequences(tokenizer.texts_to_sequences(dataset_questions), maxlen = max_seq_length)
    preds = list(malstm.predict([test_data_left, test_data_right], verbose=1).ravel())
    indexed_preds = list(zip(test_indices, preds))
    indexed_preds.sort(key=itemgetter(1), reverse=True)
    result_questions = []
    result_answers= []
    for tup in indexed_preds:
        index = tup[0]
        result_questions.append(dataset_questions[test_indices.index(index)])
        result_answers.append(test_answers[test_indices.index(index)])
    results = [(x, y, w[0], w[1]) for (x, y), w in zip(zip(result_questions, result_answers), indexed_preds)]
    with open('RESULTSQQ.txt', 'w+', encoding='utf-8') as fp:
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
   test_question = 'how to get someones mail id from their name?'
   questions_vs_answers(test_question)
   questions_vs_questions(test_question)
   

if __name__  == '__main__':
    main()