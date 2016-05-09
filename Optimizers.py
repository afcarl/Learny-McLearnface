# -*- coding: utf-8 -*-
"""
Created on Sun May 08 23:36:11 2016

@author: Alexander Weaver
"""

"""
A collection of first order optimization methods.
In general, these methods will take an array of parameters (theta)
and an array of their derivatives with respect to some function, L
This will attempt to find parameters which minimize L, iteratively.  
These methods must be iterated an indefinitenumber of times in order to achieve the best result.
Each method will take a dictionary of options, which contains settings specific to that method
"""

"""
Stochastic gradient descent.
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
This method takes one option:
    'learning_rate': a real number >0, overall step size
"""
def sgd(theta, dtheta, options):
    try:
        learning_rate = options['learning_rate']
    except KeyError:
        raise MissingOptionException('Optimization method is missing a required option.')
    theta -= learning_rate * dtheta
    return theta

class MissingOptionException(Exception):
    pass