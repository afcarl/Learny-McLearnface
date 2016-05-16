# -*- coding: utf-8 -*-
"""
Created on Mon May 16 04:11:02 2016

@author: Alexander Weaver
"""

import numpy as np

"""
Performs the hyperbolic tangent activation function elementwise on its input array 
Forward propagating this object performs the function on an input array and returns the result, saving necessary values
Back propagating this object  uses the saved values to return the derivative of the inputs before the operation
"""

class TanhLayer(object):
    
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = in_dim
        
    def forward(self, x):
        self.cache_tanh = np.tanh(x)
        return self.cache_tanh
        
    def backward(self, dout):
        return (1 - (self.cache_tanh * self.cache_tanh)) * dout