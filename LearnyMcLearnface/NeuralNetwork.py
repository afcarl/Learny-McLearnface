# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:34:21 2016

@author: Alexander Weaver
"""

import numpy as np
from . import Layers as layers
from . import Utils as utils

class NeuralNetwork(object):
    
    """
    Initializes a neural network.
    Takes a dictionary of initialization options.
    """
    def __init__(self, options):
        self.input_dim = options['input_dim']
        self.data_type = options.setdefault('data_type', np.float32)
        self.init_scheme = options.setdefault('init_scheme', 'xavier')
        self.layers = []
        self.num_layers = 0
    
    """
    Adds a layer to the neural network.
    The layer must be of a valid type, and is associated with a dictionary.
    If the layer has any special options or hyperparameters, these are indicated in the dictionary.
    Otherwise, the dictionary is empty.
    """
    def add_layer(self, layer_type, params):
        if not self.layers:
            in_dim = self.input_dim
        else:
            in_dim = self.layers[-1].out_dim
        if 'weight_scale' in params:
            weight_scale = params['weight_scale']
        elif self.init_scheme == 'xavier':
            weight_scale = 1./np.sqrt(in_dim)
        if layer_type == 'SoftmaxLoss':
            layer = layers.SoftmaxLossLayer(in_dim)
            self.layers.append(layer)
        elif layer_type == 'SVMLoss':
            layer = layers.SVMLossLayer(in_dim)
            self.layers.append(layer)
        elif layer_type == 'Affine':
            layer = layers.AffineLayer(in_dim, params['neurons'], weight_scale, self.data_type)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'Batchnorm':
            layer = layers.BatchnormLayer(in_dim, params['decay'], self.data_type)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'Dropout':
            if 'seed' in params:
                layer = layers.DropoutLayer(in_dim, params['dropout_param'], seed=params['seed'])
            else:
                layer = layers.DropoutLayer(in_dim, params['dropout_param'])
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'PReLU':
            layer = layers.PReLULayer(in_dim, self.data_type)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'ReLU':
            layer = layers.ReLULayer(in_dim)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'Sigmoid':
            layer = layers.SigmoidLayer(in_dim)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'Tanh':
            layer = layers.TanhLayer(in_dim)
            self.layers.append(layer)
            self.num_layers += 1
        else:
            raise InvalidLayerException('Invalid layer: ' + layer_type)
            
    """
    Performs forward propagation on the network, pushing a tensor through each layer in sequence.
    Does not perform final layer classification.
    """
    def forward(self, X, train=False):
        X = X.astype(self.data_type)
        forward_tensor = X
        for layer in self.layers:
            if layer == self.layers[-1]:
                return forward_tensor
            if isinstance(layer, layers.DropoutLayer) or isinstance(layer, layers.BatchnormLayer) and train:
                forward_tensor = layer.forward_train(forward_tensor)
            else:
                forward_tensor = layer.forward(forward_tensor)
            
    """
    Performs forward propagation, and performs final layer classification.
    Returns an NxC matrix of class scores per given example.
    """
    def classify(self, X):
        X = X.astype(self.data_type)
        scores = self.forward(X)
        return self.layers[-1].evaluate(scores)
        
    """
    Given a set of training examples and their corresponding scores, performs forward propagation and then
    returns the final layer classifier loss and the derivative of that loss function.
    """
    def loss(self, X, y, reg_param=0.0):
        X = X.astype(self.data_type)
        scores = self.forward(X, train=True)
        loss, dx = self.layers[-1].loss(scores, y)
        squared_sum = 0.0
        for layer in self.layers:
            if isinstance(layer, layers.AffineLayer):
                squared_sum += np.sum(layer.W * layer.W)
        loss += 0.5 * reg_param * squared_sum
        return loss, dx
        
    """
    Takes a set of training examples and corresponding scores.
    Performs forward propagation, executes the final layer classifier loss function.
    Then, performs backpropagation on the network and saves intermediate derivatives to the respective layers.
    Returns the classifier loss and its derivative for progress reporting purposes.
    """
    def backward(self, X, y, reg_param=0.0):
        X = X.astype(self.data_type)
        loss, dx = self.loss(X, y, reg_param)
        for layer in reversed(self.layers):
            if layer == self.layers[-1]: 
                continue
            dx = layer.backward(dx)
            if isinstance(layer, layers.AffineLayer):
                layer.dW += reg_param * layer.W
        return loss, dx
                
            
            
class InvalidLayerException(Exception):
    pass