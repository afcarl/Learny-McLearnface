# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:54:11 2016

@author: Alexander Weaver
"""

"""
Performs an affine (fully connected) operation on its input
An affine layer with out_dim neurons takes a data array of size Nx(in_dim), x
and returns a linearly transformed Nx(out_dim) data array
The transformation result, z, is determined by a (in_dim)x(out_dim) weight matrix, W,  and
a (out_dim) bias vector, b.  The transformation of any one data point (one row in x) is given by:
    z = Wx + b
Constructing this object initializes the parameters following a gaussian random distribution with 
standard deviation given by weight_scale.  
Forward propagating this object performs the affine transformation on the given array, X.
Backpropagating this object returns the derivatives of x, W, and b with respect to the final output of
the network.
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