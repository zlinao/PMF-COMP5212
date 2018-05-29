#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:15:05 2018

@author: lin
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
import types
class PMF(BaseEstimator, ClassifierMixin):
    """ An example classifier which implements a 1-NN algorithm.
    Parameters
    ----------
    demo_param : str, optional
        A parameter used for demonstation of how to pass and store paramters.
    Attributes
    ----------
    X_ : array, shape = [n_samples, n_features]
        The input passed during :meth:`fit`
    y_ : array, shape = [n_samples]
        The labels passed during :meth:`fit`
    """
    def __init__(self, lambda_u=0.1, lambda_v=0.1, K=2 ,lr=0.2, lr_decay=0.8 ,bsz=500 , maxepoch=20, user_num=944, movie_num=1683, random_state=0):
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v
        self.K = K
        self.lr = lr
        self.lr_decay = lr_decay
        self.bsz = bsz
        self.maxepoch = maxepoch
        self.user_num = user_num
        self.movie_num = movie_num
        self.seed = random_state
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        self.mean = np.mean(y)
        X, y = check_X_y(X, y)
        #initial U,V
        X = np.array(X,dtype='int32')
        np.random.seed(self.seed)
        U = 0.1*np.random.randn(self.user_num,self.K)
        V = 0.1*np.random.randn(self.movie_num,self.K)
        
        self.batch_num = int(len(X)/self.bsz)
        count = 0
        best_rmse=100
        for epoch in range(self.maxepoch):
            shuffled_order = np.arange(X.shape[0]) 
            np.random.shuffle(shuffled_order) 
            
            #early stop
            movie_id = X[:,0]
            user_id = X[:,1]
            predi = np.sum(np.multiply(U[user_id],V[movie_id]),axis=1)+self.mean
            rmse = np.linalg.norm(y - predi)/np.sqrt(len(y))
            print('epoch{}: lambda_u={}, lambda_v={}, K={} ,lr={:.2f},rmse={:.2f}'.format(epoch,self.lambda_u,self.lambda_v,self.K,self.lr,rmse))
            if rmse<best_rmse:
                best_rmse=rmse
            else:
                count+=1
            if count == 2:
                break
            
            
            for batch in range(self.batch_num):
                batch_idx = np.arange(batch*self.bsz,(batch+1)*self.bsz)
                batch_movieid = np.array(X[shuffled_order[batch_idx],0],dtype = 'int32')
                batch_userid = np.array(X[shuffled_order[batch_idx],1],dtype = 'int32')
                pred = np.sum(np.multiply(U[batch_userid],V[batch_movieid]),axis = 1) 
                #gradient
                batch_U_grad1 = np.multiply(pred - y[shuffled_order[batch_idx]]+self.mean,V[batch_movieid].T).T+self.lambda_u*U[batch_userid]
                batch_V_grad1 = np.multiply(pred - y[shuffled_order[batch_idx]]+self.mean,U[batch_userid].T).T +self.lambda_u*V[batch_movieid]
                # loop to aggreate the gradients of the same element
                batch_U_grad2 = np.zeros((self.user_num,self.K))
                batch_V_grad2 = np.zeros((self.movie_num,self.K))
                for i in range(self.bsz):
                    batch_U_grad2[batch_userid[i]] += batch_U_grad1[i]
                    batch_V_grad2[batch_movieid[i]] += batch_V_grad1[i]
                #updata
                U -= self.lr*batch_U_grad2
                V -= self.lr*batch_V_grad2
            self.lr = self.lr*self.lr_decay
            
                #print(U[:10],V[:10])
        # Store the classes seen during fit
        self.U_ = U
        self.V_ = V
        self.classes_ = unique_labels(y)
        
        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Input validation
        X = check_array(X)
        X = np.array(X,dtype='int32')
        movie_id = X[:,0]
        user_id = X[:,1]
        self.y_ = np.sum(np.multiply(self.U_[user_id],self.V_[movie_id]),axis=1)+self.mean
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_
    """
    def set_params(self, parameters):
        if isinstance(parameters, dict):
            
            self.lambda_u = parameters.get('lambda_u',0.1)
            self.lambda_v = parameters.get('lambda_v',0.1)
            self.K = parameters.get('K',2)
            self.lr = parameters.get('lr',1)
            self.bsz = parameters.get('bsz',1000)
            self.maxepoch = parameters.get('maxepoch',20)
            self.user_num = parameters.get('user_num',943)
            self.movie_num = parameters.get('movie_num',1682)
            self.seed = parameters.get('random_state',0)
    """