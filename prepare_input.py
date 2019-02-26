# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:41:52 2019

@author: peter
"""

import pandas as pd

"""
Read the given file to a dataset
The first column is of date format. This will be set as index
Then calculate the direction and add it as the last column
This last step is for validation only
The data is from yahoo. They add the Adj Close. This column is deleted.
"""
def prepareInput(filename):
    dataset_in = pd.read_csv(filename, sep= ',', index_col = 0)
    # Delete Adj Close
    dataset_in = dataset_in.drop(['Adj Close'], axis=1)
    # Add reference for validation only      
    dataset_in['direction'] = getDirection(dataset_in)     
    return dataset_in

"""
We are starting with a record (prepareInput) like
DATE       | *** | direction  
01-02.2019 | *** | 0
02-02.2019 | *** | 1
03-02.2019 | *** | 1

The goal is to train the ANN/RNN to predict the direction of the next day
So we create a dataset where we shift the direction to the record of the day before

DATE       | *** | direction  
01-02.2019 | *** | 1
02-02.2019 | *** | 1
03-02.2019 | *** | -

therefore we need to delete the last record
"""
def getDataset(filename):
    dataset = pd.DataFrame(prepareInput(filename))
    direction = getDirection(dataset)
    direction = direction.shift(-1)
    dataset['direction'] = direction
    dataset = dataset.drop(dataset.index[len(dataset)-1])
    return dataset

"""
The direction is difference of "Open" minus "Close".
If direction < 0 which means the direction is up the direction is set to 1 else 0 
"""
def getDirection(dataset):
    direction = dataset['Open'] - dataset['Close']
    direction.loc[direction > 0] = 0  
    direction.loc[direction < 0] = 1
    return direction

"""
This will return the Data as index and
the openings 
"""
def getOpen(dataset):
    openings = dataset.iloc[:, 0:1].values
    return openings

"""
This will return the Data as index and
the close values 
"""
def getClose(dataset):
    close = dataset.iloc[:, 3:4].values
    return close

# Test only 
in_ = prepareInput('GDAXI.csv')
open = getOpen(in_)
close = getClose(in_)