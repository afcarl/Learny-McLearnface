# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:13:30 2016

@author: Alexander Weaver
"""

import numpy as np

class BatchnormLayer(object):

    def __init__(self, in_dim, decay, data_type=np.float32):
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.decay = decay
        self.data_type = data_type
        self.eps = 1e-5
        self.running_mean = np.zeros(in_dim, data_type)
        self.running_var = np.zeros(in_dim, data_type)
        self.gamma = np.ones(in_dim)
        self.beta = np.zeros(in_dim)
        
    def forward_train(self, x):
        N, D = x.shape
        sample_mean = np.mean(x, axis=0).reshape(1, D)
        sample_var = np.var(x, axis=0).reshape(1, D)
        self.running_mean = self.decay * self.running_mean + (1 - self.decay) * sample_mean
        self.running_var = self.decay * self.running_var + (1 - self.decay) * sample_var
        normalized = (x - sample_mean)/np.sqrt(sample_var + self.eps)
        self.cache = (x, normalized, self.gamma, (x - sample_mean), 1/np.sqrt(sample_var + self.eps), np.sqrt(sample_var + self.eps), sample_var, self.eps)
        return self.gamma*normalized + self.beta
        
    def forward(self, x):
        normalized = (x - self.running_mean)/np.sqrt(self.running_var + self.eps)
        return self.gamma*normalized + self.beta
    
    def backward(self, dout):
        N, D = dout.shape
        x, normalized, gamma, centered_x, inverted_var, sqrt_var, var, eps = self.cache
        self.dbeta = np.sum(dout, axis=0)
        self.dgamma = np.sum(dout * normalized, axis=0)
        dnormalized = dout * gamma
        d_inverted_var = np.sum(dnormalized * centered_x, axis=0)
        d_centered_x1 = dnormalized * inverted_var
        d_sqrt_var = d_inverted_var * (-1/(sqrt_var**2))
        d_var = 0.5 * (1/np.sqrt(var + eps)) * d_sqrt_var
        d_sq = (1./N) * np.ones((N, D)) * d_var
        d_centered_x2 = 2 * centered_x * d_sq
        dx1 = d_centered_x1 + d_centered_x2
        du = -np.sum(dx1, axis=0)
        dx2 = (1./N) * np.ones((N, D)) * du
        return dx1 + dx2
    