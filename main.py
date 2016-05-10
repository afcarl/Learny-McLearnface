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

from Evaluator import *
from Trainer import *

def main():
    nn = NeuralNetwork(28*28)
    nn.add_layer('Affine', {'neurons':500, 'weight_scale':1e-2})
    nn.add_layer('ReLU', {})
    nn.add_layer('Affine', {'neurons':10, 'weight_scale':1e-2})
    nn.add_layer('SoftmaxLoss', {})
    
    test_data = np.random.randn(100, 28*28)
    test_y = np.random.randint(1, 10, 100)
    
    evaluator = Evaluator(nn, test_data)
    print evaluator.predict()[0].shape
    print evaluator.predict()[1].shape
    
    nn.backward(test_data, test_y)
    print nn.layers[0].dW.shape
    print nn.layers[0].db.shape
    
    probabilities = nn.classify(test_data)
    N, C = probabilities.shape
    predicted_classes = np.argmax(probabilities, axis=1)
    print np.mean(predicted_classes == test_y)
    
main()