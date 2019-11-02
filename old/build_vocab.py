# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 23:22:05 2018

@author: Skanda
"""

import pickle

def vocab_build():
    data = list()
    with open('final_cleaned_data.pickle', 'rb') as fp:
        data = pickle.load(fp)
    vocabulary = set()
    for item in data:
        text = item[1] + item[2]
        for word in text:
            vocabulary.add(word)
    vocabulary = list(vocabulary)
    with open('vocabulary.pickle', 'wb') as fp:
        pickle.dump(vocabulary, fp)
    del vocabulary

def inverted_index_build():
    vocabulary = list()
    data = list()
    inverted_index = dict()
    with open('vocabulary.pickle', 'rb') as fp:
        vocabulary = pickle.load(fp)
    with open('final_cleaned_data.pickle', 'rb') as fp:
        data = pickle.load(fp)
    for word in vocabulary:
        postings_list = []
        for item in data:
            text = item[1] + item[2]
            if word in text:
                postings_list.append(item[0])
        inverted_index[word] = postings_list
    with open('inverted_index.pickle', 'wb') as fp:
        pickle.dump(inverted_index, fp)
    del vocabulary
    del data
    del inverted_index
    
                
def main():
    vocab_build()
    inverted_index_build()
    
if __name__ == '__main__':
    main()
    