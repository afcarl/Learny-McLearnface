# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:55:21 2016

@author: Alexander Weaver
"""

"""
Performs the softmax loss function and its derivative on a set of class scores
Given N examples with D classes, takes an NxD array representing the class scores
Interprets the scores as unnormalized log probabilities, and performs the cross-entropy loss function
Returns the loss function and the derivatives of the scores with respect to the loss, for backpropagation
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