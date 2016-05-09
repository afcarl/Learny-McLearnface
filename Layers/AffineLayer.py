# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:54:11 2016

@author: Alexander Weaver
"""

import numpy as np

class AffineLayer(object):
    
    def __init__(self, in_dim, out_dim, weight_scale):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_scale = weight_scale
        self.W = np.random.randn(in_dim, out_dim) * weight_scale
        self.b = np.zeros(out_dim)
        
    def forward(self, x, W=None, b=None):
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        N = x.shape[0]
        reshaped_x = x.reshape(N, np.prod(x.shape[1:]))
        out = reshaped_x.dot(W) + b
        self.cache_x = x
        return out
        
    def backward(self, dout):
        x = self.cache_x
        N = x.shape[0]  
        reshaped_x = x.reshape(N, np.prod(x.shape[1:]))
        dx = dout.dot(np.transpose(self.W)).reshape(x.shape)
        dW = np.transpose(reshaped_x).dot(dout)
        db = np.sum(dout, axis=0)
        return dx, dW, db