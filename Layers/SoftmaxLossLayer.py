# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:55:21 2016

@author: Alexander Weaver
"""

import numpy as np

class SoftmaxLossLayer(object):
    
    def __init__(self, in_dim):
        self.in_dim = in_dim
        
    def loss(self, scores, y):
        N, D = scores.shape
        shifted_scores = scores - np.max(scores, axis = 1, keepdims=True)
        probabilities = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis = 1, keepdims = True)
        loss = np.sum(-np.log(probabilities[(range(N), y)])) / N
        dx = probabilities.copy()
        dx[range(N), y] -= 1
        dx /= N
        return loss, dx