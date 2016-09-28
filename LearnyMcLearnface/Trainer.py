# -*- coding: utf-8 -*-
"""
Created on Mon May 09 15:09:03 2016

@author: Alexander Weaver
"""

import numpy as np
import copy

from .Optimizers import *

from . import Layers as layers

class Trainer(object):
    
    """
REQUIRED OPTIONS:
    'update_options' : a dictionary of options for the update rule
OPTIONAL OPTIONS:
    'batch_size' : The number of examples trained at a time. Default is 128.
    'num_epochs' : The number of passes to make through the dataset.  Default is 1.
    'reg_param' : The regularization parameter lambda. Nonnegative real number, 0 corresponds to regularization "off"
    """
    def __init__(self, model, data, options):
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        try:
            self.update_options = options['update_options']
        except KeyError:
            raise MissingOptionException('Trainer is missing a required option.')
        self.batch_size = options.setdefault('batch_size', 128)
        self.num_epochs = options.setdefault('num_epochs', 1)
        self.reg_param = options.setdefault('reg_param', 0)
        
        
    def train(self):
        N = self.X_train.shape[0]
        updates_per_epoch = max(N//self.batch_size, 1)
        num_updates = updates_per_epoch * self.num_epochs
        for i in range(num_updates):
            self.update()
            if self.is_end_of_epoch(i, updates_per_epoch):
                print('Epoch', (i+1)//updates_per_epoch, 'of', self.num_epochs, 'Validation accuracy:', self.accuracy(self.X_val, self.y_val))
        
    """
    Performs a single gradient descent update
    """
    def update(self):
        X_batch, y_batch = self.get_batch()
        loss, dx = self.model.backward(X_batch, y_batch, self.reg_param)
        for layer in self.model.layers:
            if isinstance(layer, layers.AffineLayer):
                if not hasattr(layer, 'update_options_W'):
                    layer.update_options_W = copy.deepcopy(self.update_options)
                if not hasattr(layer, 'update_options_b'):
                    layer.update_options_b = copy.deepcopy(self.update_options)
                layer.W, layer.update_options_W = optimize(layer.W, layer.dW, layer.update_options_W)
                layer.b, layer.update_options_b = optimize(layer.b, layer.db, layer.update_options_b)
            if isinstance(layer, layers.PReLULayer):
                if not hasattr(layer, 'update_options'):
                    layer.update_options = copy.deepcopy(self.update_options)
                layer.W, layer.update_options = optimize(layer.W, layer.dW, layer.update_options)
            if isinstance(layer, layers.BatchnormLayer):
                if not hasattr(layer, 'update_options_gamma'):
                    layer.update_options_gamma = copy.deepcopy(self.update_options)
                if not hasattr(layer, 'update_options_beta'):
                    layer.update_options_beta = copy.deepcopy(self.update_options)
                layer.gamma, layer.update_options_gamma = optimize(layer.gamma, layer.dgamma, layer.update_options_gamma)
                layer.beta, layer.update_options_beta = optimize(layer.beta, layer.dbeta, layer.update_options_beta)
        
    """
    Samples a random minibatch of data from the training set, self.batch_size elements
    Returns the minibatch data and labels in array form
    """
    def get_batch(self):
        N = self.X_train.shape[0]
        indices = np.random.choice(N, self.batch_size)
        X_batch = self.X_train[indices]
        y_batch = self.y_train[indices]
        return X_batch, y_batch
        
    """
    Returns the percent accuracy of the network on the given dataset and labels
    """
    def accuracy(self, X, y):
        probabilities = self.model.classify(X)
        N, C = probabilities.shape
        predicted_classes = np.argmax(probabilities, axis=1)
        return np.mean(predicted_classes == y)
        
    def is_end_of_epoch(self, iteration_index, iterations_per_epoch):
        return (iteration_index+1) % iterations_per_epoch == 0
        
class MissingOptionException(Exception):
    pass