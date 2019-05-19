# -*- coding: utf-8 -*-
"""
Created on Wed May  1 19:06:21 2019

@author: peter
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


class Trainer (object):
    timestamp = 60

    """
    dataset: numpy.ndarray data provided 
    input: 
    """
    sc = MinMaxScaler(feature_range = (0, 1))
    
    def __init__(self,inputfile, x_columns_names, y_column_name):
       dataset = self.getDataset(inputfile)
       fitting_set = np.array(dataset[y_column_name].copy())
       fitting_set = np.delete(fitting_set,0,0)
       self.sc.fit(fitting_set.reshape(-1,1))
       #self.sc.fit(dataset.values.reshape(-1,1))
       input = self.initX(dataset, x_columns_names)
       self.dimension = len(x_columns_names)
       self.reference = self.initY(dataset, y_column_name)
       self.model = self.createModel(input)      
 
    """
    Inpot are the input values value for today. There is no therefore
    for the last record so we delete this
    """      
    def initX(self, dataset, x_column_names):
       training_set = dataset[x_column_names.copy()].values
       for i in range (0, len(x_column_names)):
           training_set = np.append(training_set, self.scale(training_set[:,i]), axis = 1)
       for i in range (0, len(x_column_names)):    
           training_set = np.delete(training_set,0,axis = 1)
       return np.delete(training_set,len(training_set)-1,0)
       #return training_set
   
    """
    Reference is the close value of the next day therefore the first record is deleted
    """
    def initY(self, dataset, y_column_name):
        output_set = dataset[y_column_name].copy()
        output_set = self.scale(output_set)
        training_items = len(output_set)
        y = []
        for i in range(self.timestamp, training_items):
            y.append(output_set[i, 0])
        y = np.array(y)
        return np.delete(y,0,0)
        #return y
   
    def getDataset(self,filename):
        dataset_in = pd.read_csv(filename, sep= ',')
        #dataset_in = pd.read_csv(filename, sep= ',', index_col = 0)
        dataset_in = dataset_in.drop(['Adj Close', 'Volume'], axis=1)
        dataset_in = dataset_in.dropna()  
        return dataset_in
    
    def createModel(self, input):
        training_set = input
        trainingItems = len(input)   
        X_1_train = []
        for j in range(self.timestamp, trainingItems):
            X_1_train.append(training_set[j-self.timestamp:j, :])
        X_1_train = np.array(X_1_train)
        X_1_train = np.reshape(X_1_train, (X_1_train.shape[0], X_1_train.shape[1], self.dimension))
        return X_1_train
    
    def scale(self, unscaled):
        temp = np.array(unscaled)
        scaled = self.sc.transform(temp.reshape(-1,1))
        return scaled
    
    def train(self, _layers, _units, _algorithm, _error, _epochs, _batch_size):
        regressor = Sequential()
        regressor.add(LSTM(units = _units, return_sequences = True, input_shape = (self.model.shape[1], self.dimension)))
        regressor.add(Dropout(0.2))
        for i in range (0, _layers):
            regressor.add(LSTM(units = _units, return_sequences = True))
            regressor.add(Dropout(0.2))
        regressor.add(LSTM(units = _units))
        regressor.add(Dropout(0.2))
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = _algorithm, loss = _error)
        regressor.fit(self.model, self.reference, epochs = _epochs, batch_size = _batch_size)
        self.persistModel(regressor, self.dimension, _layers, _units, _algorithm, _error, _epochs, _batch_size)
        
    def persistModel(self, regressor,_dim, _layers, _units, _algorithm, _error, _epochs, _batch_size):
        regressor_json = regressor.to_json()
        name = 'd_'+str(_dim)
        name = name + '_l_'+str(_layers)
        name = name + '_u_' +str(_units)
        #name = name + '_a_' +str(_algorithm)
        #name = name + '_er_' +str(_error)
        #name = name + '_ep_' +str(_epochs)
        #name = name + '_b_' +str(_batch_size)

        with open('regressors/'+name+'_dax_regressor.json', 'w') as json_file:
            json_file.write(regressor_json)
        regressor.save_weights('regressors/'+name+'_dax_regressor.h5')

       
tr = Trainer('./daten/offset_2013.csv',['Open','Close'],'Close')

"""
ds = tr.getDataset('./daten/offset_2019.csv')
input_ = tr.initX(ds, ['Open','Close'])
output = tr.initY(ds, ['Close'])
short = np.delete(output,0,0)
dsm = ds[['Open', 'Close']].copy().copy().as_matrix()
"""
#tr.train(3, 60, 'adam', 'mean_squared_error', 150, 32)
# tr.train(3, 60, 'adam', 'mean_squared_logarithmic_error', 200, 32) --> 2019-05-07_19-49-44_1_dax_regressor forget it
# tr.train(3, 60, 'rmsprop', 'mean_squared_error', 200, 32) 2019-05-07_22-46-13_1_dax_regressor good
#tr.train(3, 60, 'rmsprop', 'mean_squared_logarithmic_error', 150, 32)

for i in range(1,8):
    units = 40 +(i*5)
    tr.train(4, units, 'rmsprop', 'mean_squared_logarithmic_error', 150, 32)
    print('##########################################################')
    print('##########################################################')      
