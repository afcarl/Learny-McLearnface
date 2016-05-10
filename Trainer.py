# -*- coding: utf-8 -*-
"""
Created on Mon May 09 15:09:03 2016

@author: Alexander Weaver
"""

class Trainer(object):
    
    """
REQUIRED OPTIONS:
    'update_options' : a dictionary of options for the update rule
OPTIONAL OPTIONS:
    'batch_size' : The number of examples trained at a time. Default is 128.
    'num_epochs' : The number of passes to make through the dataset.  Default is 1.
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
        
        
    def train(self):
        N = self.X_train.shape[0]
        updates_per_epoch = max(N/self.batch_size, 1)
        num_updates = updates_per_epoch * self.num_epochs
        
    """
    Performs a single gradient descent update
    """
    def update(self):
        loss, dx = self.model.loss(self.X_train, self.y_train)
        
        
class MissingOptionException(Exception):
    pass