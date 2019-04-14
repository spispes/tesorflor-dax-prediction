# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:13:32 2019

@author: peter
"""

from keras.models import model_from_json

def load_regressor(name):
    reg = 'regressors/'+name+ '.json'
    weights = 'regressors/'+name+ '.h5'
    with open(reg, 'r') as f:
        regressor = model_from_json(f.read())
    regressor.load_weights(weights)
    return regressor



# evaluate loaded model on test data
#loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))