# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 15:25:50 2018

@author: Skanda
"""

import xml.etree.ElementTree as ET
from os import listdir
import pandas as pd
import random



def show(elem, indent = 0):
    '''
    given the root of an xml file, finds the children recursively
    '''
    print(' ' * indent + elem.tag)
    for child in elem.findall('*'):
        show(child, indent + 1)
        
        
def create_negative_samples(total_data, ID, question, category):
    number_of_negative_samples = 5
    negative_samples = []
    M = len(total_data)
    for i in range(number_of_negative_samples):
        x = random.randint(0, M-1)
        datapoint = total_data[x]
        if datapoint['ID'] == ID:
            i = i-1
            continue
        row = [datapoint['subject'] + datapoint['content'], ' '.join(datapoint['answers']), category, 0]
        negative_samples.append(row)
    return negative_samples
    
    
    
def create_dataset(total_data):
    positive_samples = []
    negative_samples = []
    for datapoint in total_data:
        ID = datapoint['ID']
        question = datapoint['subject'] + datapoint['content']
        answers = ' '.join(datapoint['answers'])
        category = datapoint['category']
        similarity = 1
        row = [question, answers, category, similarity]
        positive_samples.append(row)
        negative_samples.extend(create_negative_samples(total_data, ID, question, category))
    final_data = pd.DataFrame(positive_samples, columns=['Question', 'Answer', 'Category', 'Similarity'])
    df = pd.DataFrame(negative_samples, columns=['Question', 'Answer', 'Category', 'Similarity'])
    final_data = final_data.append(df)
    final_data = final_data.dropna()
    print(final_data.isnull().sum())
    final_data.to_csv('intermediate.csv', encoding='utf-8')
        
        
        
def extract_data():
    my_path = 'Dataset Yahoo small\\xml'
    tag_suffix = '{urn:yahoo:answers}'
    files = [f for f in listdir(my_path)]
    total_data = []
    for f in files:
        data = {}
        tree = ET.parse(my_path + '\\' + f)
        root = tree.getroot()
        data['ID'] = root.find(tag_suffix + 'Question').get('id')
        data['subject'] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Subject').text
        data['content'] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Content').text
        data['category'] = root.find(tag_suffix + 'Question/' + tag_suffix + 'Category').text
        answers = []
        for e in root.findall(tag_suffix + 'Question/' + tag_suffix + 'Answers/' + tag_suffix + 'Answer'):
            answers.append(e.find(tag_suffix + 'Content').text)
        data['answers'] = answers
        total_data.append(data)
    return total_data
    
    
def main():
    total_data = extract_data()
    create_dataset(total_data) 
    df = pd.read_csv('intermediate.csv')
    df = df.dropna()
    df.to_csv('dataset.csv', encoding='utf-8')
    
    


if __name__ == '__main__':
    main()
    
   

