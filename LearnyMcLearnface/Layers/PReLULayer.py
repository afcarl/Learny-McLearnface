# -*- coding: utf-8 -*-
"""
Created on Sun May 22 17:36:16 2016

@author: Alexander Weaver
"""

import numpy as np

class PReLULayer(object):
    
    def __init__(self, in_dim, data_type=np.float32):
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.data_type = data_type
        self.W = np.zeros(in_dim)
        self.W = self.W.astype(self.data_type)
        
    def forward(self, x, W=None):
        if W is None:
            W = self.W
        self.cache_x = x
        return (x * (x > 0)) + ((x * W) * (x < 0))
    
    def backward(self, dout):
        N, D = dout.shape
        self.dW = np.sum(dout * (self.cache_x * (self.cache_x <= 0)), axis=0)
        return dout * ((self.cache_x >= 0) + (self.W * self.cache_x < 0))