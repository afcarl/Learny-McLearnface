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
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def add_layer(self, layer_type, params):
        if not self.layers:
            in_dim = self.input_dim
        else:
            in_dim = self.layers[-1].out_dim
        if layer_type == 'SoftmaxLoss':
            layer = SoftmaxLossLayer(in_dim)
            self.layers.append(layer)
        elif layer_type == 'Affine':
            layer = AffineLayer(in_dim, params['neurons'], params['weight_scale'])
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
        forward_tensor = X
        for layer in self.layers:
            if layer == self.layers[-1]:
                return self.layers[-1].evaluate(forward_tensor)
            forward_tensor = layer.forward(forward_tensor)
            
class InvalidLayerException(Exception):
    pass