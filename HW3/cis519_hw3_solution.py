#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 
# #**Homework 3 : Logistic Regression**

# In[538]:


import pandas as pd
import numpy as np


# ### Logistic Regression

# In[539]:


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.0001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = initTheta
        self.costList = []
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        h_theta = self.sigmoid(np.dot(X, theta))
        h_theta = np.array(h_theta)
        cost = -(np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
        if self.regNorm is 1:
            regCost = regLambda * sum(abs(theta))
        if self.regNorm is 2:
            regCost = regLambda * ((np.linalg.norm(theta)) ** 2)
        return cost + regCost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        h_theta = self.sigmoid(np.dot(X, theta))
        n, d = X.shape
        gradient = np.zeros((d,1))
        h_theta = np.array(h_theta)
        gradient[0,0] = sum(h_theta - y) # no regularization for the x_i0
        for j in range(d-1):
            if self.regNorm is 1:
                gradient[j+1,0] = np.dot(X[:,j+1].reshape((1,n)), (h_theta - y)) + regLambda * ( theta[j+1, 0] / abs(theta[j+1, 0]) )
            if self.regNorm is 2:
                gradient[j+1,0] = np.dot(X[:,j+1].reshape((1,n)), (h_theta - y)) + regLambda * theta[j+1, 0]
        return gradient
        

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        n = len(y)
        X = X.to_numpy()
        X = np.c_[np.ones((n, 1)), X]
        
        n, d = X.shape
        y = y.to_numpy()
        y = y.reshape(n, 1)
        
        # start doing gradient descent
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1))) #np.zeros(d)
        
        
        count = 0
        self.costList.append((None, 100)) # assign a large dummy theta and assign None to cost as the starter 
        while self.hasConverged(self.theta, self.costList[count][1]):
            self.costList.append( (self.computeCost(self.theta, X, y, self.regLambda), self.theta) )
            count = count + 1
            print("Iteration: ", count, " Cost: ", self.costList[count][0], " Theta.T: ", self.theta.T)
            self.theta = self.theta - self.alpha * self.computeGradient(self.theta, X, y, self.regLambda)
            if count is self.maxNumIters:
                break
            


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        prob = self.predict_proba(X)
        label = (prob >= 0.5)
        return label

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''
        n, d = X.shape
        X = X.to_numpy()
        X = np.c_[np.ones((n, 1)), X]
        prob = pd.DataFrame(self.sigmoid(np.dot(X, self.theta)))
        return prob
    
    def hasConverged(self, theta, prevTheta):
        running_epsilon = np.linalg.norm( theta - prevTheta)
        print('The convergence now is: ', running_epsilon)
        return (running_epsilon > self.epsilon)


    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1 + np.exp(-Z))


# # Test Logistic Regression 1

# In[479]:


# Test script for training a logistic regressiom model
#
# This code should run successfully without changes if your implementation is correct
#
from numpy import loadtxt, ones, zeros, where
import numpy as np
from pylab import plot,legend,show,where,scatter,xlabel, ylabel,linspace,contour,title
import matplotlib.pyplot as plt

def test_logreg1():
    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data1.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape
    
    # # Standardize features
    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    Xstandardized = pd.DataFrame(standardizer.fit_transform(X))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    regLambda = 0.00000001
    regNorm = 2
    logregModel = LogisticRegression(regLambda = regLambda, regNorm = regNorm)
    logregModel.fit(Xstandardized,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.savefig('Logreg Decision Boundary with lambda ' + str(regLambda) + ' under L' + str(regNorm))
    plt.show()
    

#test_logreg1()


# # Map Feature

# In[540]:


def mapFeature(X, column1, column2, maxPower = 6):
    '''
    Maps the two specified input features to quadratic features. Does not standardize any features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the maxPower polynomial
        
    Arguments:
        X is an n-by-d Pandas data frame, where d > 2
        column1 is the string specifying the column name corresponding to feature X1
        column2 is the string specifying the column name corresponding to feature X2
    Returns:
        an n-by-d2 Pandas data frame, where each row represents the original features augmented with the new features of the corresponding instance
    '''
    X_1_poly = pd.DataFrame(X.iloc[:, 0])
    X_2_poly = pd.DataFrame(X.iloc[:, 1])
    i, j = X.shape
    for d in range(maxPower-1):
            X_1_poly = pd.concat([X_1_poly, X.iloc[:, 0] * X_1_poly.iloc[:, d]], axis = 1)
            X_2_poly = pd.concat([X_2_poly, X.iloc[:, 1] * X_2_poly.iloc[:, d]], axis = 1)
    X_1_poly = pd.concat([pd.DataFrame(np.ones((i, 1))), X_1_poly], axis = 1)
    
    instance, degree_with_one = X_1_poly.shape
    degree_without_one = degree_with_one - 1
    
    X_mapped = pd.DataFrame()
    X_mapped = pd.concat([X_mapped, X_1_poly], axis = 1)
    
    for d in range(degree_with_one):
        for complement in range(degree_without_one - d):
            X_mapped = pd.concat([X_mapped, X_1_poly.iloc[:, d] * X_2_poly.iloc[:, complement]], axis = 1)
    return X_mapped


# # Test Logistic Regression 2

# In[594]:


from numpy import loadtxt, ones, zeros, where
import numpy as np
from pylab import plot,legend,show,where,scatter,xlabel, ylabel,linspace,contour,title
import matplotlib.pyplot as plt

def test_logreg2():

    polyPower = 6

    # load the data
    filepath = "http://www.seas.upenn.edu/~cis519/spring2020/data/hw3-data2.csv"
    df = pd.read_csv(filepath, header=None)

    X = df[df.columns[0:2]]
    y = df[df.columns[2]]

    n,d = X.shape

    # map features into a higher dimensional feature space
    Xaug = mapFeature(X.copy(), X.columns[0], X.columns[1], polyPower)

    # # Standardize features
    from sklearn.preprocessing import StandardScaler
    standardizer = StandardScaler()
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # compute mean and stdev on training set for standardization
    
    # train logistic regression
    logregModel = LogisticRegressionAdagrad(regLambda = 0.00000001, regNorm=2, epsilon=0.00000001, maxNumIters = 10000)
    logregModel.fit(Xaug,y)
    
    # Plot the decision boundary
    h = .02  # step size in the mesh
    x_min = X[X.columns[0]].min() - .5
    x_max = X[X.columns[0]].max() + .5
    y_min = X[X.columns[1]].min() - .5
    y_max = X[X.columns[1]].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    allPoints = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()])
    allPoints = mapFeature(allPoints, allPoints.columns[0], allPoints.columns[1], polyPower)
    allPoints = pd.DataFrame(standardizer.transform(allPoints))
    Xaug = pd.DataFrame(standardizer.fit_transform(Xaug))  # standardize data
    
    Z = logregModel.predict(allPoints)
    Z = np.asmatrix(Z.to_numpy())

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot the training points
    plt.scatter(X[X.columns[0]], X[X.columns[1]], c=y.ravel(), edgecolors='k', cmap=plt.cm.Paired)
    
    # Configure the plot display
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    plt.savefig('')
    plt.show()


    print(str(Z.min()) + " " + str(Z.max()))

#test_logreg2()


# # Logistic Regression with Adagrad

# In[593]:


class LogisticRegressionAdagrad:

    def __init__(self, alpha = 0.05, regLambda=0.00001, regNorm=2, epsilon=0.00001, maxNumIters = 10000, initTheta = None):
        '''
        Constructor
        Arguments:
        	alpha is the learning rate
        	regLambda is the regularization parameter
        	regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
        	epsilon is the convergence parameter
        	maxNumIters is the maximum number of iterations to run
          initTheta is the initial theta value. This is an optional argument
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.regNorm = regNorm
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = initTheta
        self.costList = []
        self.Xi = 1e-05

    
    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        d = len(X)
        h_theta = self.sigmoid(np.dot(X.reshape((1, d)), theta))
        #h_theta = np.array(h_theta)
        cost = -(np.dot(y.T, np.log(h_theta)) + np.dot((1 - y).T, np.log(1 - h_theta)))
        if self.regNorm is 1:
            regCost = regLambda * sum(abs(theta))
        if self.regNorm is 2:
            regCost = regLambda * ((np.linalg.norm(theta)) ** 2)
        return cost + regCost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-by-1 numpy matrix
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        #h_theta = np.array(h_theta)
        d = len(X)
        h_theta = self.sigmoid(np.dot(X.reshape((1, d)), theta))
        gradient = np.zeros((d,1))
        # compute special case, the first column with x_i0 all being 1
        gradient[0, 0] = h_theta - y # no regularization for the x_i0
        # start compute the rest of columns
        for j in range(d-1):
            if self.regNorm is 1:
                gradient[j+1, 0] = (h_theta - y) * X[j+1] + regLambda * ( theta[j+1]/ abs(theta[j+1] ) )
            if self.regNorm is 2:
                gradient[j+1, 0] = (h_theta - y) * X[j+1] + regLambda * theta[j+1]        
        return gradient
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d Pandas data frame
            y is an n-by-1 Pandas data frame
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before fit() is called.
        '''
        n = len(y)
        X = X.to_numpy()
        X = np.c_[np.ones((n, 1)), X]
        
        n, d = X.shape
        y = y.to_numpy()
        y = y.reshape(n, 1)
        
        # start doing gradient descent
        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))  #np.zeros(d)
        
        
        count = 0
        G = np.zeros((d, 1))
        self.alpha = np.ones((d,1)) * self.alpha
        self.costList.append((None, 100)) # assign a large dummy theta and assign None to cost as the starter 
        allData = np.c_[X, y]
        prev_theta = np.matrix(np.ones((d,1))) # initialize a dummy theta value
        for rep in range(self.maxNumIters):
            np.random.shuffle(allData)
            X = allData[:,:-1]
            y = allData[:,-1]
            for i in range(n):
                G = G + ((self.computeGradient(self.theta, X[i,:], y[i], self.regLambda)).reshape((d,1)))**2
                G = sum(G)
                alpha_t = self.alpha / (np.sqrt(G) + self.Xi) 
                prev_theta = self.theta # store the previous theta values
                self.theta = self.theta - alpha_t * self.computeGradient(self.theta, X[i,:], y[i], self.regLambda)
                if self.hasConverged(self.theta, prev_theta):
                    break
            count = count + 1
            print('Iteration ', count, ' theta: ', self.theta.T)
            
            


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 dimensional Pandas data frame of the predictions
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict() is called.
        '''
        prob = self.predict_proba(X)
        label = (prob >= 0.5)
        return label

    def predict_proba(self, X):
        '''
        Used the model to predict the class probability for each instance in X
        Arguments:
            X is a n-by-d Pandas data frame
        Returns:
            an n-by-1 Pandas data frame of the class probabilities
        Note:
            Don't assume that X contains the x_i0 = 1 constant feature.
            Standardization should be optionally done before predict_proba() is called.
        '''
        n, d = X.shape
        X = X.to_numpy()
        X = np.c_[np.ones((n, 1)), X]
        prob = pd.DataFrame(self.sigmoid(np.dot(X, self.theta)))
        return prob

    def hasConverged(self, theta, prevTheta):
        running_epsilon = np.linalg.norm( self.theta - prevTheta)
        print('The convergence now is: ', running_epsilon)
        return (running_epsilon < self.epsilon)


    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1 + np.exp(-Z))

   


# In[ ]:





# In[ ]:




