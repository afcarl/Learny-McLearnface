# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:34:21 2016

@author: Alexander Weaver
"""

import numpy as np
from . import Layers as layers
from . import Utils as utils

class NeuralNetwork(object):
    
    
    def __init__(self, options):
        self.input_dim = options['input_dim']
        self.data_type = options.setdefault('data_type', np.float32)
        self.init_scheme = options.setdefault('init_scheme', 'xavier')
        self.layers = []
        self.num_layers = 0
    
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
        elif layer_type == 'Dropout':
            if 'seed' in params:
                layer = layers.DropoutLayer(in_dim, params['dropout_param'], seed=params['seed'])
            else:
                layer = layers.DropoutLayer(in_dim, params['dropout_param'])
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
            
    def forward(self, X, train=False):
        X = X.astype(self.data_type)
        forward_tensor = X
        for layer in self.layers:
            if layer == self.layers[-1]:
                return forward_tensor
            if isinstance(layer, layers.DropoutLayer) and train:
                forward_tensor = layer.forward_train(forward_tensor)
            else:
                forward_tensor = layer.forward(forward_tensor)
            
    def classify(self, X):
        X = X.astype(self.data_type)
        scores = self.forward(X)
        return self.layers[-1].evaluate(scores)
        
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