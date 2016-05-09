# -*- coding: utf-8 -*-
"""
Created on Fri May 06 17:56:07 2016

@author: Alexander Weaver
"""

import numpy as np
from NeuralNetwork import *
from Utils.NumericalGradient import *

from Layers.SoftmaxLossLayer import *
from Layers.AffineLayer import *
from Layers.ReLULayer import *

def main():
    nn = NeuralNetwork(28*28)
    nn.add_layer('Affine', {'neurons':500, 'weight_scale':1e-2})
    nn.add_layer('ReLU', {})
    nn.add_layer('Affine', {'neurons':10, 'weight_scale':1e-2})
    nn.add_layer('SoftmaxLoss', {})
    
    test_data = np.random.randn(100, 28*28)
    test_y = np.random.randint(1, 10, 100)
    
    affine1 = nn.layers[0].forward(test_data)
    relu = nn.layers[1].forward(affine1)
    affine2 = nn.layers[2].forward(relu)
    loss, grads_b_softmax = nn.layers[3].loss(affine2, test_y)
    print 'Loss:', loss 
    
main()