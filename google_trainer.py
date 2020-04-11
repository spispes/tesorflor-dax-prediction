# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:13:54 2019

@author: peter
"""

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
from tensorflow.python.data import Dataset



class Trainer (object):
    tf.logging.set_verbosity(tf.logging.ERROR)
    pd.options.display.max_rows = 10
    pd.options.display.float_format = '{:.1f}'.format
    
    
    def __init__(self,inputfile):
       self.dataset = pd.read_csv(inputfile, sep= ',')
       self.dataset = self.dataset.drop(['Adj Close', 'Volume'], axis=1)
       self.dataset = self.dataset.dropna()
       self.dataset.reindex(np.random.permutation(self.dataset.index))
       #self.features = self.getfeaturesDict()
       #self.targets = self.getTarget()
       #self.dataset = self.getDataset()
       
    def getFeaturesDict(self,input_features):
        feature_data = self.dataset[[input_features]].astype('float32')
        #feature_data.drop(feature_data.tail(1).index,inplace=True)
        feature_dictionary = {key:np.array(value) for key,value in dict(feature_data).items()} 
        """
        input_feature = 'Close'
        feature_data = self.dataset[[input_feature]].astype('float32')
        feature_data.drop(feature_data.tail(1).index,inplace=True)
        feature_dictionary.update({key:np.array(value) for key,value in dict(feature_data).items()})
        """
        #feature_dictionary['Close_1'] = feature_dictionary.pop('Close')
        #feature_dictionary['Open-1'] = feature_dictionary.pop('Open')
        
        """
        input_feature = 'Open'
        feature_data = self.dataset[[input_feature]].astype('float32')
        feature_data.drop(feature_data.head(1).index,inplace=True)
        feature_dictionary.update({key:np.array(value) for key,value in dict(feature_data).items()})        
        """
        return feature_dictionary
        
    def getTarget(self,input_targets):
        target_data = self.dataset[input_targets].astype('float32')
        #target_data.drop(target_data.head(1).index,inplace=True)
        return target_data       
       
    def getDataset(self,input_features,input_targets,batch_size=1,num_epochs=None): 
        features = self.getFeaturesDict(input_features)
        targets = self.getTarget(input_targets)
        ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    
    def describeDataset(self):
        return self.dataset.describe()
    
    def input_fn(self,input_features, input_targets, batch_size=1, num_epochs=None):
        #ds = self.getDataset()
        #features = {key:np.array(value) for key,value in dict(self.getFeaturesDict).items()}  
        features, labels = self.getDataset(input_features, input_targets, batch_size, num_epochs).make_one_shot_iterator().get_next()
        return features, labels
    
    def train_model(self, input_features, input_targets, learning_rate, steps, batch_size):
        """Trains a linear regression model.     
            Args:
            learning_rate: A `float`, the learning rate.
            steps: A non-zero `int`, the total number of training steps. A training step
              consists of a forward and backward pass using a single batch.
            batch_size: A non-zero `int`, the batch size.
            input_feature: A `string` specifying a column from `california_housing_dataframe`
              to use as input feature.
              
              Returns:
            A Pandas `DataFrame` containing targets and the corresponding predictions done
            after training the model.
        """
        periods = 5
        steps_per_period = steps / periods
        
        
        #my_feature_data = self.getFeaturesDict()
        targets = self.getTarget(input_targets)
        
          # Create input functions.
        training_input_fn = lambda: self.input_fn(input_features, input_targets, batch_size=batch_size)
        predict_training_input_fn = lambda: self.input_fn(input_features, input_targets, num_epochs=1)
          
        # Create feature columns.
        feature_columns = [tf.feature_column.numeric_column(input_features)]
        
        # Create a linear regressor object.
        my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_columns,
        optimizer=my_optimizer
        )
        
        # Set up to plot the state of our model's line each period.
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.title("Learned Line by Period")
        plt.ylabel(input_targets)
        plt.xlabel(input_features)
        sample = self.dataset.sample(n=30)
        plt.scatter(sample[input_features], sample[input_targets])
        colors = [cm.coolwarm(x) for x in np.linspace(-1, 1, periods)]
        
        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("RMSE (on training data):")
        root_mean_squared_errors = []
        for period in range (0, periods):
            # Train the model, starting from the prior state.
            linear_regressor.train(input_fn=training_input_fn,steps=steps_per_period)
            # Take a break and compute predictions.
            predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
            predictions = np.array([item['predictions'][0] for item in predictions])
            
            # Compute loss.
            root_mean_squared_error = math.sqrt(
              metrics.mean_squared_error(predictions, targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            root_mean_squared_errors.append(root_mean_squared_error)
            # Finally, track the weights and biases over time.
            # Apply some math to ensure that the data and line are plotted neatly.
            y_extents = np.array([0, sample[input_features].max()])
            
            weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_features)[0]
            bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
            
            x_extents = (y_extents - bias) / weight
            x_extents = np.maximum(np.minimum(x_extents, sample[input_features].max()),sample[input_features].min())
            y_extents = weight * x_extents + bias
            plt.plot(x_extents, y_extents, color=colors[period]) 
        print("Model training finished.")
        
        # Output a graph of loss metrics over periods.
        plt.subplot(1, 2, 2)
        plt.ylabel('RMSE')
        plt.xlabel('Periods')
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(root_mean_squared_errors)
        
        # Create a table with calibration data.
        calibration_data = pd.DataFrame()
        calibration_data["predictions"] = pd.Series(predictions)
        calibration_data["targets"] = pd.Series(targets)
        display.display(calibration_data.describe())
        
        print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)
          
        return calibration_data

tr = Trainer('./daten/training.csv')
#dataset_in = tr.getDataset('Close','Close')
#statistic = tr.describeDataset()

#print(tr.input_fn(batch_size=5))
#features = tr.getFeaturesDict()
#targets = tr.getTarget()
training = tr.train_model('Open','Close', learning_rate=0.005, steps=500, batch_size=15)
feature = tr.getFeaturesDict('Open')
target = tr.getTarget('Close')