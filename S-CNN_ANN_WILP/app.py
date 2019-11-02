# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:09:34 2019

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

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers import Bidirectional
from keras.preprocessing.sequence import pad_sequences
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
#tokenizer.fit_on_texts(documents)
with open('tokenizer.pickle', 'rb') as fp:
    tokenizer = pickle.load(fp)
embedding_dim = 100
embeddings = None
with open('embeddings.pickle', 'rb') as fp:
    embeddings = pickle.load(fp)
gc.collect()
            
max_seq_length = max(max([len(x.split(' ')) for x in questions]), max([len(x.split(' ')) for x in answers]))
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2)

train_left = tokenizer.texts_to_sequences(list(X_train['Questions']))
train_right = tokenizer.texts_to_sequences(list(X_train['Answers']))
validation_left = tokenizer.texts_to_sequences(list(X_validation['Questions']))
validation_right = tokenizer.texts_to_sequences(list(X_validation['Answers']))
X_train = {'left': train_left, 'right': train_right}
X_validation = {'left': validation_left, 'right': validation_right}

def convert_to_matrix(lst):
    new_lst = []
    for sub_lst in lst:
        new_sub_lst = []
        for value in sub_lst:
            new_sub_lst.append(embeddings[value])
        new_lst.append(new_sub_lst)
    return np.array(new_lst)
    

# Convert labels to their numpy representations
y_train = y_train.values
y_validation = y_validation.values


# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)
    
X_train['left'] = convert_to_matrix(X_train['left'])
X_train['right'] = convert_to_matrix(X_train['right'])
X_validation['left'] = convert_to_matrix(X_validation['left'])
X_validation['right'] = convert_to_matrix(X_validation['right'])

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(y_train)

X_train['left'] = np.expand_dims(X_train['left'], axis=3)
X_train['right'] = np.expand_dims(X_train['right'], axis=3)
X_validation['left'] = np.expand_dims(X_validation['left'], axis=3)
X_validation['right'] = np.expand_dims(X_validation['right'], axis=3)
# Model variables
batch_size = 64
n_epoch = 10


input_shape = (125, 100, 1)
left_input = Input(input_shape)
right_input = Input(input_shape)
#build convnet to use in each siamese 'leg'
convnet = Sequential()
convnet.add(Conv2D(32,(5,5),activation='relu',input_shape=input_shape))
convnet.add(MaxPooling2D())
convnet.add(Conv2D(32,(5,5),activation='relu'))
convnet.add(MaxPooling2D())
convnet.add(Flatten())
convnet.add(Dense(10,activation='sigmoid'))

encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

merged = concatenate([encoded_l, encoded_r])
merged = BatchNormalization()(merged)
merged = Dropout(0.20)(merged)
merged = Dense(30, activation='relu')(merged)
merged = BatchNormalization()(merged)
merged = Dropout(0.20)(merged)
preds = Dense(1, activation='sigmoid')(merged)
model = Model(inputs=[left_input, right_input], outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Start training
training_start_time = time()

model_trained = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=n_epoch,
                            shuffle=True, validation_data=([X_validation['left'], X_validation['right']], y_validation))

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


model.save('SCNN-ANN.h5')



def questions_vs_answers(test_question, category):
    max_seq_length = 125
    with open('tokenizer.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    model = load_model('SCNN-ANN.h5')
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
    test_data_left = convert_to_matrix(test_data_left)
    test_data_right = convert_to_matrix(test_data_right)
    test_data_left = np.expand_dims(test_data_left, axis=3)
    test_data_right = np.expand_dims(test_data_right, axis=3)
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
    model = load_model('SCNN-ANN.h5')
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
    test_data_left = convert_to_matrix(test_data_left)
    test_data_right = convert_to_matrix(test_data_right)
    test_data_left = np.expand_dims(test_data_left, axis=3)
    test_data_right = np.expand_dims(test_data_right, axis=3)
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
   test_question = 'not able to login'
   category = 'general'
   questions_vs_answers(test_question, category)
   questions_vs_questions(test_question, category)
   

if __name__  == '__main__':
    main()