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
    
    data = {
        'X_train' : test_data,
        'y_train' : test_y,
        'X_val' : test_data,
        'y_val' : test_y
    }
    
    update_opts = {
        'update_rule' : 'sgd',
        'learning_rate' : 1
    }
    
    opts = {
        'update_options' : update_opts,
        'reg_param' : 0
    }    
    
    trainer = Trainer(nn, data, opts)
    
    loss, _ = nn.backward(test_data, test_y, 0)
    print 'Initial loss: ', loss
    
    updates = 5
    for i in range(updates):
        trainer.update()
        loss, _ = nn.backward(test_data, test_y, 0)
        print 'Loss after ', i+1, ' update(s): ', loss
    
    probabilities = nn.classify(test_data)
    N, C = probabilities.shape
    predicted_classes = np.argmax(probabilities, axis=1)
    print np.mean(predicted_classes == test_y)
    
    
    
main()