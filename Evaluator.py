# -*- coding: utf-8 -*-
"""
Created on Mon May 09 14:48:01 2016

@author: Alexander Weaver
"""

"""
Single evaluator instance for a neural network
Takes a model and a dataset to be evaluated
Let N be the number of examples, C be the number of classes, and D be the dimension of the inputs
Data is expected to be an NxC matrix where the rows are input data points
The output is an NxC matrix of probabilities, where the ith row and jth column is the classification
probability for the Nth example in the Cth class
When predicting, a pair of vectors is returned
The first vector indicates the predicted class for each example
The second vector indicates the percent confidence for each prediction, correspondingly
"""

from NeuralNetwork import *

class Evaluator(object):
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    def get_probabilities(self):
        probabilities = self.model.classify(self.data)
        return probabilities
        
    def predict(self):
        probabilities = self.get_probabilities()
        N, C = probabilities.shape
        predicted_classes = np.argmax(probabilities, axis=1)
        confidences = probabilities[(range(N), predicted_classes)]
        return predicted_classes, confidences
        