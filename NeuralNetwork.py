# -*- coding: utf-8 -*-
"""
Created on Fri May 06 14:34:21 2016

@author: Alexander Weaver
"""

from Layers.SoftmaxLossLayer import *
from Layers.AffineLayer import *
from Layers.ReLULayer import *
from Layers.SigmoidLayer import *

class NeuralNetwork(object):
    
    layers = []
    num_layers = 0
    
    def __init__(self, input_dim, data_type=np.float32):
        self.input_dim = input_dim
        self.data_type = data_type
    
    def add_layer(self, layer_type, params):
        if not self.layers:
            in_dim = self.input_dim
        else:
            in_dim = self.layers[-1].out_dim
        if layer_type == 'SoftmaxLoss':
            layer = SoftmaxLossLayer(in_dim)
            self.layers.append(layer)
        elif layer_type == 'Affine':
            layer = AffineLayer(in_dim, params['neurons'], params['weight_scale'], self.data_type)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'ReLU':
            layer = ReLULayer(in_dim)
            self.layers.append(layer)
            self.num_layers += 1
        elif layer_type == 'Sigmoid':
            layer = SigmoidLayer(in_dim)
            self.layers.append(layer)
            self.num_layers += 1
        else:
            raise InvalidLayerException('Invalid layer: ' + layer_type)
            
    def forward(self, X):
        X = X.astype(self.data_type)
        forward_tensor = X
        for layer in self.layers:
            if layer == self.layers[-1]:
                return forward_tensor
            forward_tensor = layer.forward(forward_tensor)
            
    def classify(self, X):
        X = X.astype(self.data_type)
        scores = self.forward(X)
        return self.layers[-1].evaluate(scores)
        
    def loss(self, X, y, reg_param=0.0):
        X = X.astype(self.data_type)
        scores = self.forward(X)
        loss, dx = self.layers[-1].loss(scores, y)
        squared_sum = 0.0
        for layer in self.layers:
            if type(layer).__name__ == 'AffineLayer': #TODO: upon upgrade to Python 3, "if layer is AffineLayer"
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
            if type(layer).__name__ == 'AffineLayer':   #TODO: same as above
                layer.dW += reg_param * layer.W
        return loss, dx
                
            
            
class InvalidLayerException(Exception):
    pass