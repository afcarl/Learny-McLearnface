# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:22:13 2016

@author: Alexander Weaver
"""

import numpy as np

class SVMLossLayer(object):
    
    def __init__(self, in_dim):
        self.in_dim = in_dim
        
    def loss(self, scores, y):
        N, D = scores.shape
        margin = scores - scores[np.arange(N), y].reshape(N, 1) + 1.0   #Fj - Fy + 1
        margin[np.arange(N), y] = 0
        margin[margin < 0] = 0	#max(0, Fj - Fy + 1)
        loss = np.sum(margin/N)
        num_positive = np.sum(margin > 0, axis=1)
        dx = np.zeros(scores.shape)
        dx[margin > 0] = 1
        dx[np.arange(N), y] -= num_positive
        dx /= N
        return loss, dx
        
        
    def evaluate(self, scores):
        return scores