#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 
# #**Homework 4 : Adaboost and the Challenge**

# In[ ]:


import pandas as pd
import numpy as np


# # Adaboost-SAMME

# In[328]:


import numpy as np
import math
from sklearn import tree

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor

        Class Fields 
        clfs : List object containing individual DecisionTree classifiers, in order of creation during boosting
        betas : List of beta values, in order of creation during boosting
        '''

        self.clfs = None  # keep the class fields, and be sure to keep them updated during boosting
        self.betas = None
        
        #TODO
        self.numBoostingIters = numBoostingIters
        self.maxTreeDepth = maxTreeDepth
        self.K = None
        self.classes = None
        



    def fit(self, X, y, random_state=None):
        '''
        Trains the model. 
        Be sure to initialize all individual Decision trees with the provided random_state value if provided.
        
        Arguments:
            X is an n-by-d Pandas Data Frame
            y is an n-by-1 Pandas Data Frame
            random_seed is an optional integer value
        '''
        #TODO
        X = X.to_numpy()
        y = y.to_numpy()
        n, d = X.shape
        y = y.reshape((n, 1))
        weights = np.zeros((n, 1)) + 1 / n
        
        #y = np.where(y == 0, -1, y)
        
        # Initialization
        if self.betas is None:
            self.betas = []
        if self.clfs is None:
            self.clfs = []
        if self.classes is None:
            self.classes = np.unique(y)
        if self.K is None:
            self.K = len(self.classes)
            self.classes.reshape((self.K, 1))
        
        
        for iter in range(self.numBoostingIters):
            clf = self.get_weightedDT(X, y, weights, self.maxTreeDepth, random_state)
            y_train = clf.predict(X).reshape((n, 1))
            #------for dummy autograder-------
            #y_train = np.where(y_train == 0, -1, y_train)
            #---------------------------------
            epsilon = (weights[ (~(y_train == y)).reshape((n, 1)) ]).sum() # weighted training error
            beta = 0.5 * (np.log( (1 - epsilon) / epsilon ) + np.log(self.K - 1)) # beta, the importance for current model
            self.betas.append(beta)
            self.clfs.append(clf)
            accuracy_array = (y_train == y).astype('int32').reshape((n, 1))
            # incorrect prediction -> -1, correct prediction -> 1
            sign_array = np.where( accuracy_array == 0, -1, accuracy_array)
            weights = weights * np.exp(-beta * sign_array) # update the weight
            weights = weights / sum(weights) # normalize the weights
    
    def get_weightedDT(self, X, y, weight, maxTreeDepth, random_seed):
        """
        Inputs:
            X is an n-by-d numpy array
            y is an n-by-1 numpy array
            weight is an n-by-1 numpy array
        Outputs:
            A Decision Tree Model with weighted bootstrap sampling
        """
        
        # Resampling n instanced based on the weight
        from sklearn import tree
        n, d = X.shape
        weight = weight.reshape((n,))
        clf = tree.DecisionTreeClassifier(max_depth = maxTreeDepth, random_state = random_seed)
        clf = clf.fit(X, y, sample_weight = weight)
        return clf
        

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is an n-by-d Pandas Data Frame
        Returns:
            an n-by-1 Pandas Data Frame of the predictions
        '''
        #TODO
        n, d = X.shape
        votes_array = np.zeros((n, self.K))
        for index, clf in enumerate(self.clfs):
            cur_predict = clf.predict(X).reshape((n, 1))
            # fill out the votes array
            votes_array = votes_array + (cur_predict == self.classes).astype('int32') * self.betas[index]
        y_predict = self.classes[np.argmax(votes_array, axis = 1)] # find the index associated with max votes
        y_predict = pd.DataFrame( np.where(y_predict == -1, 0, y_predict) )
        
        return y_predict
            
            
            
        