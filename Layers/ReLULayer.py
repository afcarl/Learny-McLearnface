# -*- coding: utf-8 -*-
"""
Created on Sat May 07 00:48:47 2016

@author: Alexander Weaver
"""

import numpy as np

class ReLULayer(object):
    
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = in_dim
    
    def forward(self, x):
        self.cache_x = x
        return x * (x > 0)
    
    def backward(self, dout):
        return dout * (self.cache_x >= 0)