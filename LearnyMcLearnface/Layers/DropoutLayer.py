# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:22:12 2016

@author: Alexander Weaver
"""

import numpy as np

class DropoutLayer(object):
    
    def __init__(self, in_dim, p, seed=None):
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.p = p
        if seed is not None:
            self.seed = seed
    
    def forward_train(self, x):
        try:
            np.random.seed(seed=self.seed)
        except AttributeError:
            seed = None
        mask = (np.random.rand(*x.shape) < self.p) / self.p
        self.cache_mask = mask
        return x * mask
        
    def forward(self, x):
        return x
        
    
    def backward(self, dout):
        return self.cache_mask * dout