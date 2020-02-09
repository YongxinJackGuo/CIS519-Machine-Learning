#!/usr/bin/env python
# coding: utf-8

# # CIS 419/519 Homework 1
# 
# Name: Yongxin Guo
# 
# Pennkey: yongxin
# 
# PennID: 68201122

# In[62]:


import random 
import numpy as np
import pandas as pd
random.seed(42)  # don't change this line


# In[63]:


# Load all data tables
# baseDir = "" ## TODO: insert path to data file
# df = pd.read_csv(baseDir+'hw1-NHANES-diabetes-train.csv')

# # Output debugging info
# print(df.shape)
# df.head()


# In[64]:


# Print information about the dataset
# print('Percentage of instances with missing features:')
# print(df.isnull().sum(axis=0)/df.shape[0])
# print('Class information:')
# print(df['DIABETIC'].value_counts())


# In[65]:


# # test section
# # display(df.describe())
# dftest = pd.DataFrame({'A':[1,1,1,1],'B':[0,1,0,0], 'C':[0,1,1,1]})
# dftest1 = pd.DataFrame({'A':[1,1,0,0],'B':[0,1,0,1], 'C':[0,1,1,1]})
# display(dftest)
# display(dftest1)
# display(pd.concat([dftest,dftest1], axis = 1))


# ## **Preprocessing**
# 
# The first key step in any data modeling task is cleaning your dataset. Explore your dataset and figure out what sort of preprocessing is required. Good preprocessing can make or break your final model. So choose wisely.
# 
# Some of the preprocessing steps that you can consider are :
# 
# 
# *   One-hot encoding of variables
# *   Missing value imputation
# *   Removing outliers
# *   Converting binary features into 0-1 representation
# 
# 
# Feel free to reuse code you've already written in HW 0.
# 
# 
# 
# 
# 

# In[66]:

#
# #------------------------------drop the missing value features---------------------
# # declare a missing ratio criteria for trimming the data
# trim_crt = 0.25
# # drop the column with a lot of missing values
# drop_col_bol = df.isna().mean() > trim_crt
# # number of columns with missing value larger than the criterion
# print('The total number with missing values larger than', trim_crt, 'to be dropped is: ', sum(drop_col_bol))
# # total numbers of columns
# print('\nThe total number of columns is: ', df.shape[1])
# # delete the columns with missing values larger than the criterion
# df = df.loc[:, df.columns[~drop_col_bol]]
# # check if we delete all
# print('\nAfter trimming, now the number of columns above the criteria is: ', sum(df.isna().mean() > trim_crt))
# # print the total column number
# print('\nThe total column now is: ', df.shape[1])


# In[67]:

#
# #------------------------------One-hot Encoding---------------------------------------
# # Find out if there is any non-numerical features
# non_num_count = (df.dtypes == object).sum() # there is 31 non-numerical features
# print('There are ', non_num_count, ' non-numerical features needed to be replaced with OHE')
# non_num_pos = (df.dtypes == object) # a boolean series indicating which col is object
# non_num_col = df.columns[non_num_pos] # get the non-numerical feature column
# onehots_col = non_num_col + "_" # define the prefix
# df_onehots = pd.get_dummies(df[non_num_col], prefix = onehots_col) # get the onehots col
# df.drop(non_num_col, axis = 1, inplace = True) # drop the non-numerical col
# df = pd.concat([df, df_onehots], axis = 1) # concatenate the onehots and original dataframe
# print("\n\n\n\nHere is the onehot version of data below: ")
# display(df.head())


# In[68]:


#--------------------------choose the columns we are interested in-------------------
# use correlation matrix to select the features
# corr_matrix = df.corr(method = 'pearson')
# # choose the correlation value larger than 0.25
# df = df.loc[:, (abs(corr_matrix.loc[:, 'DIABETIC']) > 0.25)]
# df = df.drop(['DIQ010'], axis = 1)

# # Find out if there is any non-numerical features
# non_num_count = (df.dtypes == object).sum() # there is 31 non-numerical features
# print('There are ', non_num_count, ' non-numerical features needed to be replaced with OHE')
# non_num_pos = (df.dtypes == object) # a boolean series indicating which col is object
# non_num_col = df.columns[non_num_pos] # get the non-numerical feature column

# chosen_features = ['RIDAGEYR', 'BMXWAIST', 'BMXHT', 'LBXTC', 'BMXLEG', 'BMXWT', 'BMXBMI',
#                   'RIDRETH1', 'BPQ020', 'ALQ120Q', 'DMDEDUC2', 'RIAGENDR', 'INDFMPIR',
#                   'LBXPLTSI', 'LBXWBCSI', 'LBXLYPCT	', 'LBXMOPCT', 'LBXNEPCT', 'LBXEOPCT', 
#                    'LBXBAPCT', 'LBDLYMNO', 'LBDMONO', 'LBDNENO', 'LBDEONO', 'LBDBANO', 
#                    'LBXRBCSI', 'LBXHGB', 'LBXHCT', 'LBXMCVSI', 'LBXMCHSI', 'LBXMC', 
#                    'LBXRDW', 'LBXPLTSI', 'LBXMPSI', 'PHQ020', 'PHQ030', 'PHQ040', 
#                    'PHQ050', 'PHQ060', 'PHAFSTHR.x', 'PHAFSTMN.x' ,'DIABETIC']
# df = pd.concat([df.loc[:, chosen_features], df[non_num_col]], axis = 1)


# In[69]:


#----------------------------Replace the rest of NaN with means--------------------------
# df_without_onehots = df.drop(df_onehots.columns, axis = 1)
# df_with_onehots = df[df_onehots.columns]
# df = pd.concat([df_without_onehots.fillna(df_without_onehots.mean()), df_with_onehots], axis = 1)
# df = df.fillna(df.mean())
#display(df.head())


# In[70]:


#---------------------------------Outlier Elimination-----------------------------------
# drop the outliers with values outside of m +/- 3*s.t.d 
# outlier_bol_upper = (df_without_onehots <= (df_without_onehots.mean() + 10 * df_without_onehots.std())).astype('int')
# outlier_bol_lower = (df_without_onehots >= (df_without_onehots.mean() - 10 * df_without_onehots.std())).astype('int')
# outlier_bol = np.logical_and(outlier_bol_upper, outlier_bol_lower)
# # max((~outlier_bol).sum()) # check what is the maximum number of outliers in any of these columns
# # sum((~outlier_bol).sum() == 8140) # check if there are columns with all being outliers (wrong)
# print('Originally the total number of instances is: ', outlier_bol.shape[0])
# outlier_bol = outlier_bol.all(axis = 1)
# print('\nAfter eliminating the outliers, the number of instances left is: ', sum(outlier_bol))
# # trim the dataframe
# df = df.loc[df.index[outlier_bol],:].reset_index(drop = True)
# display(df.head())


# In[71]:


# split the feature and the result
# X = df.drop(['DIABETIC'], axis = 1)
# y = df['DIABETIC']


# ## **Modeling**
# 
# In this section, you are tasked with building a Decision Tree classifier to predict whether or not a patient has diabetes. The overall goal of this exercise is to investigate the dataset and develop features that would improve your model performance.
# 
# To help with this process, we have provided the structure for two helper functions. These functions will help in tuning your model as well as validating your model's performance.
# 
# Complete these two functions.
# 
# 

# In[72]:


# from sklearn.model_selection import StratifiedKFold
# dftest2 = pd.DataFrame({'A':[1,2,3,4,5,5,5,6,7],'B':[0,1,0,1,0,1,1,0,1]})
# skf = StratifiedKFold(n_splits=4, random_state = None)
# dftest2_A = dftest2.drop(['B'], axis = 1)
# dftest2_B = dftest2['B']
# display(dftest2_A)
# for train_index, test_index in skf.split(dftest2_A, dftest2_B):
#     print("Train: ", train_index, "Test: ", test_index)

def cross_validated_accuracy(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    """
   Args:
        DecisionTreeClassifier: An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")
        X: Input features
        y: Labels
        num_trials: Number of trials to run of cross validation
        num_folds: Number of folds (the "k" in "k-folds")
        random_seed: Seed for uniform execution (Do not change this) 

    Returns:
        cvScore: The mean accuracy of the cross-validation experiment

    Notes:
        1. You may NOT use the cross-validation functions provided by Sklearn
    """
    ## TODO ##
    # Method 1 using RepeatedStratifiedKFold (prohibited )
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = num_trials,
                                   random_state = random_seed)
    scores = np.zeros(num_trials * num_folds) # intialize a score array with 0 entries
    # loop through all the trials(repetitions) and all the folds. 
    # Two for loops nested together in fact
    # the dataset gets shuffled before each trial/repetition
    count = 0
    for train_index, test_index in rskf.split(X, y):
        # get the x_train and x_test
        X_train, X_test = X.loc[train_index, :], X.loc[test_index, :]
        # get the y_train and y_test
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        # Model the tree
        clf = DecisionTreeClassifier.fit(X_train, y_train) 
        # prediction
        y_predict = clf.predict(X_test)
        # calculate the accuracy
        scores[count] = (y_test == y_predict).mean()
        count += 1 # update the counter
    
    cvScore = scores.mean() # get the mean accuracy
    # print('The score array is: ', scores)
    print('\nThe CV estimate of test error (Unpruned): %0.2f (+/- %0.2f)' % (1-cvScore, cvScore.std()*2))
    print('\nThe mean accuracy of the cross-validation is %0.2f: ' % cvScore)

    #================Method 2=====================
    # # intialize a score array with 0 entries
    # scores = []
    # cvScore = 0; # initialize
    # for i in range(num_trials):
    #     # concatenate two dataframes together before shuffling
    #     combined_xy = pd.concat([X, y], axis = 1)
    #     # shuffle with fixed random state
    #     combined_xy = combined_xy.sample(frac = 1, replace = False, random_state = random_seed)
    #     # split X and y again after shuffling
    #     X = combined_xy.drop(combined_xy.columns[-1], axis=1)
    #     y = combined_xy[combined_xy.columns[-1]]
    #     # create a index array for accessing "moving" test data
    #     mov_test_indices = [0] # first index must be 0
    #     # since the number of samples may not be divisble by number of folds
    #     # so the first sample should be the quotient plus the remainder
    #     # follwing the equation: (X mod n) + (x // n). should also -1 due to index
    #     remainder = (combined_xy.shape[0] % num_folds)
    #     quotient = (combined_xy.shape[0] // num_folds)
    #     mov_test_indices.append(remainder + quotient - 1)
    #     # the rest (n-1) folds have samples equaling to quotient computed above
    #     mov_test_indices = mov_test_indices + [quotient] * (num_folds - 1)
    #     # cumsum all the indices to get rid of the summing later on
    #     mov_test_indices = list(np.cumsum(mov_test_indices))
    #     for j in range(num_folds):
    #         test_upper_bound = mov_test_indices[j]
    #         test_lower_bound = mov_test_indices[j + 1]
    #         X_test = X.iloc[test_upper_bound: test_lower_bound, :]
    #         y_test = y.iloc[test_upper_bound: test_lower_bound]
    #         # concatenate the rest of X and y data as the train data
    #         X_train = pd.concat([X.iloc[0: test_upper_bound, :], X.iloc[test_lower_bound:, :]],axis = 0)
    #         y_train = pd.concat([y.iloc[0: test_upper_bound], y.iloc[test_lower_bound:]],axis = 0)
    #         # Model the tree
    #         clf = DecisionTreeClassifier.fit(X_train, y_train)
    #         y_predict = clf.predict(X_test)
    #         # calculate the accuracy
    #         scores.append((y_test == y_predict).mean())
    #
    # # get the mean accuracy
    # cvScore = np.asarray(scores).mean()
    # print('cv Score matrix: ')
    # print(scores)
    # print('\nThe CV estimate of test error (Unpruned): ', 1-cvScore, ' +/- ', cvScore.std()*2)
    # print('\nThe mean accuracy of the cross-validation is %0.2f: ' % cvScore)
    #
    return cvScore

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion = 'entropy', ccp_alpha = 0.01)
# num_trials, num_folds, random_seed = 10, 10, 10
# cvScore = cross_validated_accuracy(clf, X, y, num_trials, num_folds, random_seed)
# print('\nThe CV score is: ' , cvScore)


# In[47]:


def automatic_dt_pruning(DecisionTreeClassifier, X, y, num_trials, num_folds, random_seed):
    random.seed(random_seed)
    """
    Returns the pruning parameter (i.e., ccp_alpha) with the highest cross-validated accuracy
      Args:
            DecisionTreeClassifier  : An Sklearn DecisionTreeClassifier (e.g., created by "tree.DecisionTreeClassifier(criterion='entropy')")      
            X (Pandas.DataFrame)    : Input Features
            y (Pandas.Series)       : Labels
            num_trials              : Number of trials to run of cross validation
            num_folds               : Number of folds for cross validation (The "k" in "k-folds") 
            random_seed             : Seed for uniform execution (Do not change this)

        Returns:
            ccp_alpha : Tuned pruning paramter with highest cross-validated accuracy

        Notes:
            1. Don't change any other Decision Tree Classifier parameters other than ccp_alpha
            2. Use the cross_validated_accuracy function you implemented to find the cross-validated accuracy
    """
  ## TODO ##
    # greater value the ccp_alpha is, it increases the nodes being pruned
    # so let's start the ccp_alpha at 0.
    step_size = 0.01
    ccp_value = 0
    accuracy_list = []
    ccp_list = []
    clf = DecisionTreeClassifier
    tracker = 0;
    stop_threshold = 200
    while True:
        clf.set_params(ccp_alpha = ccp_value)
        accuracy_list.append(cross_validated_accuracy(clf, X, y, num_trials, num_folds, random_seed))
        ccp_value += step_size
        ccp_list.append(ccp_value)
        if accuracy_list[tracker] < accuracy_list[tracker - 1]:
            break
        if tracker == stop_threshold: # if it takes too long 
            break
        tracker += 1
        print('==================', tracker, '=======================')
        
    print('The accuracy list is: ', accuracy_list)
    # get the last/largest ccp_value as the best ccp_alpha 
    # since we want to pruned the tree as much as we can
    ccp_alpha = ccp_list[-1]
    return ccp_alpha

# from sklearn import tree
# clf = tree.DecisionTreeClassifier(criterion = 'entropy')
# automatic_dt_pruning(clf, X, y, 10,10,10)
  


# ## **Tuning and Testing**
# 
# With the helper functions and your processed dataset, build a Decision Tree classifier to classify Diabetic patients and tune it to maximize model performance.
# 
# Once you are done with your modeling process, test your model on the test dataset and output your predictions in a file titled "cis519_hw1_predictions.csv", with one row per prediction.

# In[ ]:


## TODO ##

