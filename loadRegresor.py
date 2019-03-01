# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:13:32 2019

@author: peter
"""

from keras.models import model_from_json


# load json and create model
json_file = open('regressor.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("regressor.h5")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))