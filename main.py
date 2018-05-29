#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 16:58:39 2018

@author: lin
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV
from model import PMF
import warnings
warnings.simplefilter('ignore')
import pickle
#from dataloader import loaddata
def RMSE(ground_truth, predictions):
    n = len(ground_truth)
    rmse = np.linalg.norm(ground_truth - predictions)/np.sqrt(n)
    return rmse

loss  = make_scorer(RMSE, greater_is_better=False)
#train_arr, test_arr = loaddata()
def load_dense():
    with open('train_data.pkl', 'rb') as f:
        train_arr = pickle.load(f, encoding='utf-8')
    with open('test_data.pkl', 'rb') as f:
        test_arr = pickle.load(f, encoding='utf-8')
    return train_arr, test_arr
def load_sparse():
    with open('train_data.pkl', 'rb') as f:
        test_arr = pickle.load(f, encoding='utf-8')
    with open('test_data.pkl', 'rb') as f:
        train_arr = pickle.load(f, encoding='utf-8')
    return train_arr, test_arr
def find_lambda(train_arr, test_arr):
    parameters = {'lambda_u':[0.1,1,10,100],'lambda_v':[0.1,1,10,100]}
    #parameters = {'lambda_u':[0.1],'lambda_v':[0.1],'lr':[0.2,0.05],'lr_decay':[0.8,0.5]}
    clf = GridSearchCV(PMF(), parameters, scoring=loss, cv=5)
    clf.fit(train_arr[:,0:2],train_arr[:,2])
    print('result report:',clf.cv_results_['mean_test_score'])
    print("optimal parameters:",clf.best_params_)
    print("best_loss:",-clf.best_score_)
    return clf.best_params_['lambda_u'],clf.best_params_['lambda_v']
def find_K(lambda_u,lambda_v,train_arr, test_arr):
    parameters = {'K':[1,2,3,4,5]}
    clf = GridSearchCV(PMF(lambda_u,lambda_v), parameters, scoring=loss, cv=5)
    clf.fit(train_arr[:,0:2],train_arr[:,2])
    print('result report:',clf.cv_results_['mean_test_score'])
    print("optimal parameters:",clf.best_params_)
    print("best_loss:",-clf.best_score_)
    return clf.best_params_['K'], -clf.best_score_
def test(lambda_u,lambda_v,K,train_arr, test_arr):
    clf = PMF(lambda_u,lambda_v,K)
    clf.fit(train_arr[:,0:2],train_arr[:,2])
    pred = clf.predict(test_arr[:,0:2])
    rmse = RMSE(test_arr[:,2],pred)
    return rmse
if __name__=='__main__':
    train_arr, test_arr = load_dense()
    lambda_u_dense,lambda_v_dense = find_lambda(train_arr, test_arr)
    K_dense, rmse_dense_val = find_K(lambda_u_dense,lambda_v_dense,train_arr, test_arr)
    rmse_dense_test = test(lambda_u_dense,lambda_v_dense,K_dense,train_arr, test_arr)
    train_arr, test_arr = load_sparse()
    lambda_u_sparse,lambda_v_sparse = find_lambda(train_arr, test_arr)
    K_sparse, rmse_sparse_val = find_K(lambda_u_sparse,lambda_v_sparse,train_arr, test_arr)
    rmse_sparse_test = test(lambda_u_sparse,lambda_v_sparse,K_sparse,train_arr, test_arr)
    print('optimal parameters for dense data: lambda_u={}, lambda_v={}, K={}, rmse_val={},rmse_test={}'.format(lambda_u_dense,lambda_v_dense,K_dense, rmse_dense_val,rmse_dense_test))
    print('optimal parameters for sparse data: lambda_u={}, lambda_v={}, K={}, rmse_val={},rmse_test={}'.format(lambda_u_sparse,lambda_v_sparse,K_sparse, rmse_sparse_val,rmse_sparse_test))