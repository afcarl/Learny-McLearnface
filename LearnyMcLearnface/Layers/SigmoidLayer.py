# -*- coding: utf-8 -*-
"""
Created on Sun May 08 20:14:20 2016

@author: Alexander Weaver
"""

"""
Performs the Sigmoid activation function elementwise on its input array 
    Sigmoid(x) = 1/(1 + e^(-x))
Forward propagating this object performs the function on an input array and returns the result, saving necessary values
Back propagating this object  uses the saved values to return the derivative of the inputs before the operation
"""

import numpy as np

class SigmoidLayer(object):
    
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = in_dim
        
    def forward(self, x):
        self.cache_x = x
        return self.sigmoid(x)
        
    def backward(self, dout):
        return (self.sigmoid(self.cache_x) * (1. - self.sigmoid(self.cache_x))) * dout
        
    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))