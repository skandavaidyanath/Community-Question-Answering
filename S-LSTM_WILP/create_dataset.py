# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:54:31 2019

@author: Skanda
"""

import pandas as pd
import random

def create_negative_samples(df, final_data, sheet, flag):
    num_rows = df.shape[0]
    num_neg_samples = 6
    negative_samples = []
    headers = list(final_data)
    if 'Expected Results' in headers:
        headers.remove('Expected Results')
    for index, row in df.iterrows():
        question = df.iloc[index,0]
        for i in range(num_neg_samples):
            neg_row = random.randint(0, num_rows-1)
            neg_answer = df.iloc[neg_row, 1]
            if flag == 2:
                negative_samples.append([question, neg_answer, 'general', 0])
            else:
                negative_samples.append([question, neg_answer, sheet, 0])
    df2 = pd.DataFrame(negative_samples, columns = headers)
    final_data = final_data.append(df2)
    return final_data
    
            
def read_data(dataset, sheets, final_data, flag):
    if flag == 1:
        xls = pd.ExcelFile(dataset)
        for sheet in sheets:
            df = pd.read_excel(xls, sheet)
            df = df.iloc[:, 2:4]
            df['category'] = sheet
            df['similarity'] = 1
            final_data = final_data.append(df)
            final_data = create_negative_samples(df, final_data, sheet, flag)
    else:
        xls = pd.ExcelFile(dataset)
        for sheet in sheets:
            df = pd.read_excel(xls, sheet)
            if sheet == '89-Questions(02-05-2018)':
                df = df.iloc[:, 1:4]
                df.drop(['Actual Results'], inplace=True, axis=1)
                df.rename(columns={'Expected Results':'Actual Results'}, inplace=True)
                df['category'] = 'general'
                df['similarity'] = 1
            else:
                df = df.iloc[:, 1:3]
                df['category'] = 'general'
                df['similarity'] = 1
            final_data = final_data.append(df)
            final_data = create_negative_samples(df, final_data, sheet, flag)
    return final_data

    
def main():
    dataset_1 = 'Dataset WILP/Software Question database.xlsx'
    dataset_2 = 'Dataset WILP/Chatbot-Results-QA.xlsx'
    sheets_1 = ['Flownex', 'Z-heat', 'Z-cast', 'J-octa', 'Flowvision', 'V-CNC', 'Afdex', 'Franc3D', 'creo']
    sheets_2 = ['89-Questions(02-05-2018)', '1-100 (30-04-2018)', '101-200 (30-04-2018)', '201-300 (02-05-2018)', '301-400 (02-05-2018)', '401-500 (02-05-2018)', '501-600 (03-05-2018)']
    final_data = pd.DataFrame()
    final_data = read_data(dataset_1, sheets_1, final_data, 1)
    final_data.rename(columns={'Unnamed: 2':'Questions', 'Unnamed: 3': 'Actual Results'}, inplace=True)
    final_data = read_data(dataset_2, sheets_2, final_data, 2)
    final_data.rename(columns={'Actual Results': 'Answers'}, inplace = True)
    final_data = final_data.dropna()
    final_data.to_csv('Dataset WILP/dataset.csv', encoding='utf-8')
    
    
if __name__ == '__main__':
    main()
    

