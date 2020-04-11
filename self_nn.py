# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:45:00 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

#Training data
x = np.array([ [0, 1], [1, 0], [1, 1],[0, 0] ])
#label
y = np.array([ [1], [1], [0], [0]])

"""
    x*(Whx) + b --> h
x1  h
x2  h   o --> y_hat
    h
    h
"""

num_input = 2
num_hidden = 5
num_output = 1

Wxh = np.random.randn(num_input,num_hidden)
bh = np.zeros((1,num_hidden))

Why = np.random.randn (num_hidden,num_output)
by = np.zeros((1,num_output))

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def forward_prop(x,Wxh,Why):
    z1 = np.dot(x,Wxh) + bh
    a1 = sigmoid(z1)
    z2 = np.dot(a1,Why) + by
    y_hat = sigmoid(z2)
    
    return z1,a1,z2,y_hat

def backword_prop(y_hat, z1, a1, z2):
    delta2 = np.multiply(-(y-y_hat),sigmoid_derivative(z2))
    dJ_dWhy = np.dot(a1.T, delta2)
    delta1 = np.dot(delta2,Why.T)*sigmoid_derivative(z1)
    dJ_dWxh = np.dot(x.T, delta1) 

    return dJ_dWxh, dJ_dWhy

def cost_function(y, y_hat):
    J = 0.5*sum((y-y_hat)**2)
    
    return J

alpha = 0.01
num_iterations = 5000

cost = []

for i in range(num_iterations):
    
    #perform forward propagation and predict output
    z1,a1,z2,y_hat = forward_prop(x,Wxh,Why)
    
    #perform backward propagation and calculate gradients
    dJ_dWxh, dJ_dWhy = backword_prop(y_hat, z1, a1, z2)
        
    #update the weights
    Wxh = Wxh -alpha * dJ_dWxh
    Why = Why -alpha * dJ_dWhy
    
    #compute cost
    c = cost_function(y, y_hat)
    
    #store the cost
    cost.append(c)
    
plt.grid()
plt.plot(range(num_iterations),cost)

plt.title('Cost Function')
plt.xlabel('Training Iterations')
plt.ylabel('Cost')

def tanh(x):
    numerator = 1-np.exp(-2*x)
    denominator = 1+np.exp(-2*x)
    return numerator/denominator

def ReLU(x):
    if x<0:
        return 0
    else:
        return x

def ReLU2(x):
    if x<0:
        return -1
    else:
        return 1

values = []
for i in range(-50,50):
    x = ReLU2(i/5)
    #x = sigmoid(i/10)
    #x = tanh(i/10)
    #x = ReLU(i/10)
    #x = (1)/(100-i)
    values.append(x)
plt.grid()
plt.plot(range(-50, 50),values)    
    

