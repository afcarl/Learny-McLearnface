# -*- coding: utf-8 -*-
"""
Created on Sun May 08 23:36:11 2016

@author: Alexander Weaver
"""

import numpy as np

"""
A collection of update rules for first order optimization methods.
In general, these methods will take an array of parameters (theta)
and an array of their derivatives with respect to some function, L
This will attempt to find parameters which minimize L, iteratively.  
These methods must be iterated an indefinite number of times in order to achieve the best result.
Each method will take a dictionary of options, which contains settings specific to that method
"""

"""
Optimization method wrapper function. Selects the correct method based on the given options.
"""
def optimize(theta, dtheta, options):
    update_rule = options.setdefault('update_rule', 'sgd')
    if update_rule == 'sgd':
        return sgd(theta, dtheta, options)
    if update_rule == 'sgd_m':
        return sgd_m(theta, dtheta, options)
    if update_rule == 'rmsprop':
        return rmsprop(theta, dtheta, options)
    if update_rule == 'adam':
        return adam(theta, dtheta, options)
    else:
        raise ValueError("The given update rule was not recognized.")

"""
Stochastic gradient descent.
https://en.wikipedia.org/wiki/Stochastic_gradient_descent
REQUIRED OPTIONS:
    'learning_rate': a real number >0, overall step size
"""
def sgd(theta, dtheta, options):
    try:
        learning_rate = options['learning_rate']
    except KeyError:
        raise MissingOptionException('Optimization method is missing a required option.')
    theta -= learning_rate * dtheta
    return theta, options

"""
Stochastic gradient descent with momentum.
REQUIRED OPTIONS:
    'learning_rate' : a real number >0, overall step size
    'momentum' : real number >=0 and <= 1, decay rate of momentum.
"""
def sgd_m(theta, dtheta, options):
    try:
        learning_rate = options['learning_rate']
        momentum = options['momentum']
    except KeyError:
        raise MissingOptionException('Optimization method is missing a required option.')
    velocity = options.get('velocity', np.zeros_like(theta))
    velocity = momentum * velocity - learning_rate * dtheta
    options['velocity'] = velocity
    return theta + velocity, options
    
"""
Root mean square propagation
REQUIRED OPTIONS:
    'learning_rate' : a real number >0, overall step size
    'decay_rate' : real number >0 and <1, the decay of the gradient cache
"""
def rmsprop(theta, dtheta, options):
    try:
        learning_rate = options['learning_rate']
        decay_rate = options['decay_rate']
    except KeyError:
        raise MissingOptionException('Optimization method is missing a required option.')
    eps = options.get('epsilon', 1e-8)
    options['epsilon'] = eps
    error = options.get('error', np.zeros_like(theta))
    error = decay_rate * error + (1 - decay_rate) * dtheta**2
    options['error'] = error
    return theta - learning_rate * dtheta / (np.sqrt(error) + eps), options
    
"""
Adaptive moment optimization
REQUIRED OPTIONS:
    'learning_rate' : a real number >0, overall step size
    'beta1' : real number >0 and <1, the decay of the average of first moment of gradient
    'beta2' : real number >0 and <1, the decay of the average of second moment of gradient
"""
def adam(theta, dtheta, options):
    try:
        learning_rate = options['learning_rate']
        beta1 = options['beta1']
        beta2 = options['beta2']
    except KeyError:
        raise MissingOptionException('Optimization method is missing a required option.')
    eps = options.get('epsilon', 1e-8)
    options['epsilon'] = eps
    m = options.get('m', np.zeros_like(theta))
    v = options.get('v', np.zeros_like(theta))
    t = options.get('t', 0)
    t += 1
    m = beta1 * m + (1 - beta1) * dtheta
    v = beta2 * v + (1 - beta2) * dtheta**2
    corrected_m = m/(1 - beta1**t)
    corrected_v = v/(1 - beta2**t)
    options['t'] = t
    options['m'] = m
    options['v'] = v
    return theta + (-learning_rate * corrected_m / (np.sqrt(corrected_v) + eps)), options

class MissingOptionException(Exception):
    pass