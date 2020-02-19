#!/usr/bin/env python
# coding: utf-8

# # CIS 519 HW 2

# In[13]:


import pandas as pd

import numpy as np
from numpy import linalg as LA
from numpy.linalg import *

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# # Linear Regression

# In[14]:


# '''
#     Linear Regression via Gradient Descent
# '''

# class LinearRegression:

#     def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
#         '''
#         Constructor
#         '''
#         self.alpha = alpha
#         self.n_iter = n_iter
#         self.theta = init_theta
#         self.JHist = None
    

#     def gradientDescent(self, X, y, theta):
#         '''
#         Fits the model via gradient descent
#         Arguments:
#             X is a n-by-d numpy matrix
#             y is an n-dimensional numpy vector
#             theta is a d-dimensional numpy vector
#         Returns:
#             the final theta found by gradient descent
#         '''
#         n,d = X.shape
#         self.JHist = []
#         for i in range(self.n_iter):
#             self.JHist.append( (self.computeCost(X, y, theta), theta) )
#             print("Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta.T: ", theta.T)
#             yhat = X*theta
#             theta = theta -  (X.T * (yhat - y)) * (self.alpha / n)
#         return theta
    

#     def computeCost(self, X, y, theta):
#         '''
#         Computes the objective function
#         Arguments:
#           X is a n-by-d numpy matrix
#           y is an n-dimensional numpy vector
#           theta is a d-dimensional numpy vector
#         Returns:
#           a scalar value of the cost  
#               ** Not returning a matrix with just one value! **
#         '''
#         n,d = X.shape
#         yhat = X*theta
#         J =  (yhat-y).T * (yhat-y) / n
#         J_scalar = J.tolist()[0][0]  # convert matrix to scalar
#         return J_scalar
    

#     def fit(self, X, y):
#         '''
#         Trains the model
#         Arguments:
#             X is a n-by-d Pandas Dataframe
#             y is an n-dimensional Pandas Series
#         '''
#         n = len(y)
#         X = X.to_numpy()
#         X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term

#         y = y.to_numpy()
#         n,d = X.shape
#         y = y.reshape(n,1)

#         if self.theta is None:
#             self.theta = np.matrix(np.zeros((d,1)))

#         self.theta = self.gradientDescent(X,y,self.theta)   


#     def predict(self, X):
#         '''
#         Used the model to predict values for each instance in X
#         Arguments:
#             X is a n-by-d Pandas DataFrame
#         Returns:
#             an n-dimensional numpy vector of the predictions
#         '''
#         X = X.to_numpy()
#         X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
#         return pd.DataFrame(X*self.theta)


# ### Test code for linear regression

# In[15]:


# def test_linreg(n_iter = 2000):
#   # load the data
#   filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-multivariateData.csv"
#   df = pd.read_csv(filepath, header=None)

#   X = df[df.columns[:-1]]
#   y = df[df.columns[-1]]

#   n,d = X.shape

#   # # Standardize features
#   from sklearn.preprocessing import StandardScaler
#   standardizer = StandardScaler()
#   X = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization

#   # # initialize the model
#   init_theta = np.matrix(np.random.randn((d+1))).T
#   alpha = 0.01

#   # # Train the model
#   lr_model = LinearRegression(init_theta = init_theta, alpha = alpha, n_iter = n_iter)
#   lr_model.fit(X,y)

#   # # Compute the closed form solution
#   X = np.asmatrix(X.to_numpy())
#   X = np.c_[np.ones((n,1)), X]     # Add a row of ones for the bias term
#   y = np.asmatrix(y.to_numpy())
#   n,d = X.shape
#   y = y.reshape(n,1)
#   thetaClosedForm = inv(X.T*X)*X.T*y
#   print("thetaClosedForm: ", thetaClosedForm.T)


# # Run the Linear Regression Test

# In[122]:


# test_linreg(2000)


# # Polynomial Regression

# In[119]:


'''
    Template for polynomial regression
'''

import numpy as np
from sklearn.linear_model import RidgeCV

#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8, tuneLambda = False, regLambdaValues = None):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda
        self.JHist = None
        self.theta = None
        self.std = None
        self.mean = None
        self.n_iter = 10000
        self.alpha = 0.005
        self.criteria = 0.0001
        self.tuneLambda = tuneLambda
        self.regLambdaValues = regLambdaValues

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        
        # initialize a initial dummy difference
        diff = 100
        n,d = X.shape
        self.JHist = []
        count = 0
        # loop until the L2 norm is less than the pre-set criteria
        while diff > self.criteria:
            self.JHist.append( (self.computeCost(X, y, theta), theta))
            print("Iteration: ", count+1, " Cost: ", self.JHist[count][0], " Theta.T: ", theta.T)
            yhat = X * theta
            theta = theta -  (X.T * (yhat - y)) * (self.alpha / n) - (self.alpha * self.regLambda * theta)
            diff = np.linalg.norm( theta - self.JHist[count][1] )
            count = count + 1
        print('The regLambda is: ', self.regLambda)
        return theta
    
    

    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** Not returning a matrix with just one value! **
        '''
        n,d = X.shape
        yhat = X*theta
        J =  (yhat-y).T * (yhat-y) / n
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar
    
    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d data frame, with each column comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not include the zero-th power.

        Arguments:
            X is an n-by-1 data frame
            degree is a positive integer
        '''
        #TODO
        X_poly = X.copy()
        for d in range(degree-1):
            X_poly = pd.concat([X_poly, X.iloc[:,0] * X_poly.iloc[:, d]], axis = 1)    
        return X_poly
    
    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 data frame
                y is an n-by-1 data frame
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling first
        '''
        #TODO
        n = len(y)
        # convert to polynomial form
        X_poly = self.polyfeatures(X, self.degree)
        # standardization
        self.mean = X_poly.mean()
        self.std = X_poly.std()
        X_poly = (X_poly - self.mean) / self.std
        # append 1s to the first column
        X_poly = X_poly.to_numpy()
        X_poly = np.c_[np.ones((n,1)), X_poly]
        
        y = y.to_numpy()
        n,d = X_poly.shape
        y = y.reshape(n,1)
        
        # check if the tuneLambda is required
        if self.tuneLambda is False:
            self.regLambda = 0
        else:
            # find the best lambda value
            clf = RidgeCV(self.regLambdaValues, cv = 4).fit(X_poly, y)
            self.regLambda = clf.alpha_
            print('clf_alpha_ is: ', clf.alpha_)
            

        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))

        self.theta = self.gradientDescent(X_poly, y, self.theta)   
        
    
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 data frame
        Returns:
            an n-by-1 data frame of the predictions
        '''
        # TODO
        n,d = X.shape
        X_copy = X.copy()
        # Convert to polynomial form
        X_copy = self.polyfeatures(X_copy, self.degree)
        # standardization
        X_copy = (X_copy - self.mean) / self.std
        X_copy = X_copy.to_numpy()
        X_copy = np.c_[np.ones((n,1)), X_copy]     # Add a row of ones for the bias term
        return pd.DataFrame(X_copy*self.theta)
        
        


# # Test Polynomial Regression

# In[120]:


# import numpy as np
# import matplotlib.pyplot as plt
#
# def test_polyreg_univariate():
#     '''
#         Test polynomial regression
#     '''
#
#     # load the data
#     filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw2-polydata.csv"
#     df = pd.read_csv(filepath, header=None)
#
#     X = df[df.columns[:-1]]
#     y = df[df.columns[-1]]
#
#     # regression with degree = d
#     d = 8
#     regLambdaValues = [1e-07, 1e-06, 1e-05, 1e-04, 0.001, 0.003, 0.006, 0.01, 0.03, 0.006, 0.1, 0.3, 0.6, 1, 3, 10]
#     tuneLambda = True
#     regLambda = 0.01
#     degree = d
#     model = PolynomialRegression(degree, regLambda, tuneLambda, regLambdaValues)
#     model.fit(X, y)
#
#     # output predictions
#     xpoints = pd.DataFrame(np.linspace(np.max(X), np.min(X), 100))
#     ypoints = model.predict(xpoints)
#
#     # plot curve
#     plt.figure()
#     plt.plot(X, y, 'rx')
#     plt.title('PolyRegression with d = '+str(d))
#     plt.plot(xpoints, ypoints, 'b-')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()


# In[121]:


# test_polyreg_univariate()


# In[ ]:




