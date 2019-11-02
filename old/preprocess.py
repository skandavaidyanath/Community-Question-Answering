# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:25:50 2018

@author: Skanda
"""

import xml.etree.ElementTree as ET
from os import listdir
import pickle
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


def show(elem, indent = 0):
    '''
    given the root of an xml file, finds the children recursively
    '''
    print(' ' * indent + elem.tag)
    for child in elem.findall('*'):
        show(child, indent + 1)
        

def preprocess(total_data):
    '''
    given a document or query string, returns a preprocessed list
    '''
    stop = stopwords.words('english') + list(string.punctuation)
    stemmer = PorterStemmer()
    category_id = dict()
    total_cleaned_data = list()
    i = 0
    for item in total_data:
        data = ['', [], [], -1, []]
        data[0] = item[0]
        data[1] = [stemmer.stem(i) for i in word_tokenize(item[1].lower()) if i not in stop]
        data[2] = [stemmer.stem(i) for i in word_tokenize(item[2].lower()) if i not in stop]
        if item[3] in category_id.keys():
            data[3] = category_id[item[3]]
        else:
            category_id[item[3]] = i
            i = i + 1
            data[3] = category_id[item[3]]
        cleaned_answers = []
        for ans in item[4]:
            cleaned_answers.append([stemmer.stem(i) for i in word_tokenize(ans.lower()) if i not in stop])
        data[4] = cleaned_answers
        total_cleaned_data.append(data)
    with open('final_cleaned_data.pickle', 'wb') as fp:
        pickle.dump(total_cleaned_data, fp)
    with open('category_id_dict.pickle', 'wb') as fp:
        pickle.dump(category_id, fp)
    del stop
    del stemmer
    del category_id
    del total_cleaned_data
    del i

def create_dataset():
    my_path = 'Dataset\\xml'
    tag_suffix = '{urn:yahoo:answers}'
    files = [f for f in listdir(my_path)]
    total_data = []
    for f in files:
        data = ['','','','',[]]
        tree = ET.parse(my_path + '\\' + f)
        root = tree.getroot()
        data[0] = root.find(tag_suffix + 'Question').get('id')
        data[1] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Subject').text
        data[2] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Content').text
        data[3] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Category').text
        answers = []
        for e in root.findall(tag_suffix + 'Question/' + tag_suffix + 'Answers/' + tag_suffix + 'Answer'):
            answers.append(e.find(tag_suffix + 'Content').text)
            data[4] = answers
        total_data.append(data)
    with open('preprocessed_data.pickle', 'wb') as fp:
        pickle.dump(total_data, fp)
    del my_path
    del tag_suffix
    del files
    del total_data
    
def main():
    create_dataset()
    total_data = list()
    with open('preprocessed_data.pickle', 'rb') as fp:
        total_data = pickle.load(fp)
    preprocess(total_data)  
    


if __name__ == '__main__':
    main()
    
   

