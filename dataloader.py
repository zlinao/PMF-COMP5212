#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 14:36:01 2018

@author: lin
"""
from tqdm import tqdm
import numpy as np
import pickle
def loaddata():
    
    data_list = []
    data = np.load('data.npz')
    
    length = len(data['item_id'])
    train_length = int(length*0.8)
    
    print('total data length:{}'.format(length))
    
    for i in tqdm(range(length)):
        
        data_list.append([int(data['item_id'][i]),int(data['user_id'][i]),float(data['rating'][i])])
    train_arr = np.array(data_list[:train_length])
    test_arr = np.array(data_list[train_length:])
    return train_arr, test_arr
train_arr, test_arr = loaddata()
with open('train_data.pkl', 'wb') as handle:
    pickle.dump(train_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('test_data.pkl', 'wb') as handle:
    pickle.dump(test_arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
    