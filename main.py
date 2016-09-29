# -*- coding: utf-8 -*-
"""
Created on Fri May 06 17:56:07 2016

@author: Alexander Weaver
"""

"""
This is an executable script used purely for experimentation.
"""

import numpy as np
import LearnyMcLearnface as lml

def main():
    
    X_train, y_train, X_test, y_test = lml.utils.get_mnist()
    
    data = {
        'X_train' : X_train,
        'y_train' : y_train,
        'X_val' : X_test,
        'y_val' : y_test
    }
    
    for i in range(10):
        pass
    
    opts = {
        'input_dim' : 28*28,
        'init_scheme' : 'xavier'
    }    
    
    nn = lml.NeuralNetwork(opts)
    nn.add_layer('Affine', {'neurons':600})
    nn.add_layer('Batchnorm', {'decay':0.9})
    nn.add_layer('PReLU', {})
    nn.add_layer('Affine', {'neurons':300})
    nn.add_layer('Batchnorm', {'decay':0.9})
    nn.add_layer('PReLU', {})
    nn.add_layer('Affine', {'neurons':10})
    nn.add_layer('SoftmaxLoss', {})
    
    opts = {
        'update_options' : {'update_rule' : 'adam', 'learning_rate' : 1e-3, 'beta1' : 0.9, 'beta2' : 0.999},
        'reg_param' : 1e-8,
        'num_epochs' : 2
    }    
    
    trainer = lml.Trainer(nn, data, opts)
    
    accuracy = trainer.accuracy(X_test, y_test)
    print('Initial model accuracy:', accuracy)
    
    trainer.train()
    
    accuracy = trainer.accuracy(X_test, y_test)
    print('Final model accuracy:', accuracy)
    
    
    
main()