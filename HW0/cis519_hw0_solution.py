#!/usr/bin/env python
# coding: utf-8

# #CIS 419/519 HW0 iPython Notebook
# 
# Complete the answers to Questions 5 and 6 by completing this notebook.
# 

# # 5.) Dynamic Programming

# In[27]:


def CompletePath(s, w, h) -> str:
    '''This function is used to escape from a room whose size is w * h.
    You are trapped in the bottom-left corner and need to cross to the
    door in the upper-right corner to escape.
    @:param s: a string representing the partial path composed of {'U', 'D', 'L', 'R', '?'}
    @:param w: an integer representing the room width
    @:param h: an integer representing the room length
    @:return path: a string that represents the completed path, with all question marks in with the correct directions
    or None if there is no path possible.
    '''

    # # TODO # #
    
    # the bottom-left corner will be a list [1, 1] and top-right will be a list [w, h], 
    def findPath(s, curRoom, s_start, trajectory):
        for index, mov in enumerate(s, start=s_start): # loop through the partial path
            if mov == "U": # going up
                curRoom[1] += 1
            elif mov == 'D': # going down
                curRoom[1] -= 1
            elif mov == 'L': # going left
                curRoom[0] -= 1
            elif mov == 'R': # going right
                curRoom[0] += 1
            else: # got a question mark ? enter the recursion section
                # recursive function to find the possible path by looping through UDLR direction
                for trial in command:
                    # substitute the question mark with the current command we make
                    path[index] = trial
                    # check if there is a solution
                    if findPath([trial] + path[index+1:], curRoom[:], index, trajectory[:]) == True: 
                        return True # return back to last layer if a solution is found
                    else:
                        pass
                return False # all four possible commands won't get to the goal, return 0 to previous layer
            if curRoom in trajectory: # check if the grid cell has been visited before
                return False # return False to last layer if it is visited
            else:
                # otherwise append the visted cell to the trajectory
                trajectory.append(curRoom.copy())
            # check if path is out of boundary
            if curRoom[0] < 1 or curRoom[0] > w or curRoom[1] < 1 or curRoom[1] > h:
                return False # return false if it is out of boundary
            
        if curRoom == goalRoom: # return true if we reach the goal room
            return True
        else: # otherwise return false
            return False
        
   
    startRoom = [1, 1] # specify the start, which is always at [1, 1]
    goalRoom = [w, h] # specify the goal room [w, h] by our definition
    command = 'UDLR' # specify the sequence of command we need to take at ? mark
    #command = 'RLDU'
    path = list(s) # create a copy of the partial path in list form
    traj_start = [[1,1]]
    
    if findPath(s, startRoom, 0, traj_start) != False:
        return "".join(path)
    else:
        return None
    pass
   
# # trial 1
# s1 = "?RDRR?UUUR" # URDRRUUUUR
# w1 = 5
# h1 = 5
# # trial 2
# s2 = "UURDD?UUR?RR" # UURDDRUURURR
# w2 = 6
# h2 = 4
# # trial 3
# s3 = "UUR?UUR?DRR?UU" # UURRUURDDRRUUU or UURUUURDDRRRUU
# w3 = 6
# h3 = 6
# # trial 4 with no solution exist
# s4 = "UUUUUUUUU?R?UU"
# w4 = 6
# h4 = 6
# # Please be noted that the possible escape path may exist multiple solutions! 
# # It depends on the sequence of the trial you choose when facing ? mark
# # for example, a command sequence of 'RLDU' and 'UDLR' will generate different soln
# # corresponding to UURRUURDDRRUUU or UURUUURDDRRRUU respectively for trial 3
# print(CompletePath(s2, w2, h2))


# # 6.) Pandas Data Manipulation

# In this section, we use the `Pandas` package to carry out 3 common data manipulation tasks :
# 
# * **Calculate missing ratios of variables**
# * **Create numerical binary variables**
# * **Convert categorical variables using one-hot encoding**
# 
# For the exercise, we will be using the Titanic dataset, the details of which can be found [here](https://www.kaggle.com/c/titanic/overview). For each of the data manipulation tasks, we have defined a skeleton for the python functions that carry out the given the manipulation. Using the function documentation, fill in the functions to implement the data manipulation.
# 

# **Dataset Link** : https://github.com/rsk2327/CIS519/blob/master/train.csv
# 
# 
# The file can be downloaded by navigating to the above link, clicking on the 'Raw' option and then saving the file.
# 
# Linux/Mac users can use the `wget` command to download the file directly. This can be done by running the following code in a Jupyter notebook cell
# 
# ```
# !wget https://github.com/rsk2327/CIS519/blob/master/train.csv
# ```
# 
# 

# In[222]:


import numpy as np
import pandas as pd


# In[223]:


# Read in the datafile using Pandas

df = pd.DataFrame(pd.read_csv('train.csv')) # read in the file # # TODO # #


# In[148]:


# !wget https://github.com/rsk2327/CIS519/blob/master/train.csv


# In[370]:


def getMissingRatio(inputDf):
    """
    Returns the percentage of missing values in each feature of the dataset.
    
    Ensure that the output dataframe has the column names: Feature, MissingPercent

    Args:
        inputDf (Pandas.DataFrame): Dataframe to be evaluated


    Returns:
        outDf (Pandas.DataFrame): Resultant dataframe with 2 columns (Feature, MissingPercent)
                                  Each row corresponds to one of the features in `inputDf`

    """
    ## TODO ##
    # a table consisting of true or false indicating if the information is missing. NaN -> true
    bolTable = inputDf.isna()
    # sum up the bolean table and divide by the length to get the missing ratio
    ratio = bolTable.sum() / len(df.index) 
    # construct the output dataframe
    outDf = pd.DataFrame({'Feature': inputDf.columns.values, 'MissingPercent': ratio.values})
    
    return outDf

#display(getMissingRatio(df))


# In[367]:


def convertToBinary(inputDf, feature):
    """
    Converts a two-value (binary) categorical feature into a numerical 0-1 representation and appends it to the dataframe
    
    Args:
        inputDf (pandas.DataFrame): Input dataframe
        variable (str) : Categorical feature which has to be converted into a numerical (0-1) representation
        
    Returns:
        outDf : Resultant dataframe with the original two-value categorical feature replaced by a numerical 0-1 feature

    """
    ## TODO ##
    # check if it is binary by getting the number of total count of the possible values 
    numOfValue = len(inputDf.loc[:, feature].dropna().unique())
    if numOfValue != 2:
        return None # return a error message
    # if it is binary, then continue
    outDf = inputDf.copy() # make a copy so that we are not modifying the original dataframe
    # get a particular value as the seed and compare it with the rest of following value resulting in bolean list, 
    # then convert to numerical datatypes. I also filter out the NaN before I get the seed.
    binaryList = (outDf.loc[:, feature] == outDf.dropna().reset_index(drop = True).loc[0, feature]).astype('int')
    binaryList = binaryList.where(~(outDf.loc[:, feature].isnull()), np.nan)
    outDf.loc[:, feature] = binaryList # replace the target feature with the numerical 0-1 representation
    return outDf

#convertToBinary(df, 'Sex')


# In[366]:


def addDummyVariables(inputDf, feature):
    """
    Create a one-hot-encoded version of a categorical feature and append it to the existing 
    dataframe.
    
    After one-hot encoding the categorical feature, ensure that the original categorical feature is dropped
    from the dataframe so that only the one-hot-encoded features are retained.
    
    For more on one-hot encoding (OHE) : https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

    Arguments:
        inputDf (Pandas.DataFrame): input dataframe
        feature (str) : Feature for which the OHE is to be performed


    Returns:
        outDf (Pandas.DataFrame): Resultant dataframe with the OHE features appended and the original feature removed

    """
    ## TODO ##
    outDf = inputDf.copy() # make a copy
    # collect all the possible values for a given feature without repetition, which then becomes a list of new features
    oneHotList = inputDf.loc[:, feature].dropna().unique() # also remove NaN
    for newFeature in oneHotList: # loop through all possible new features
        newFeatureCol = (inputDf.loc[:, feature] == newFeature).astype('int32') # compare with target feature to produce a list of binary values
        outDf[feature + '_' + newFeature] = newFeatureCol # append to the output dataframes
        
    outDf = outDf.drop(columns = feature) # remove the target feature since it was already replaced by new dummy features
    return outDf

#addDummyVariables(df,'Cabin')

