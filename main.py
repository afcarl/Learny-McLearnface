# -*- coding: utf-8 -*-
"""
Created on Fri May 06 17:56:07 2016

@author: Alexander Weaver
"""

import numpy as np
import LearnyMcLearnface as lml

def main():
    opts = {
        'init_scheme' : 'xavier'
    }    
    
    nn = lml.NeuralNetwork(700, options=opts)
    nn.add_layer('Affine', {'neurons':500})
    nn.add_layer('ReLU', {})
    nn.add_layer('Affine', {'neurons':10})
    nn.add_layer('SoftmaxLoss', {})
    
    test_data = np.random.randn(100, 700)
    test_y = np.random.randint(1, 10, 100)
    
    data = {
        'X_train' : test_data,
        'y_train' : test_y,
        'X_val' : test_data,
        'y_val' : test_y
    }
    
    opts = {
        'update_options' : {'update_rule' : 'sgd', 'learning_rate' : 1},
        'reg_param' : 0,
        'num_epochs' : 6
    }    
    
    trainer = lml.Trainer(nn, data, opts)
    
    accuracy = trainer.accuracy(test_data, test_y)
    print('Initial model accuracy:', accuracy)
    
    trainer.train()
    
    accuracy = trainer.accuracy(test_data, test_y)
    print('Final model accuracy:', accuracy)
    
    
    
main()