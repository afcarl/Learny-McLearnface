# -*- coding: utf-8 -*-
"""
Created on Fri May 06 12:16:16 2016

@author: Alexander Weaver
"""

"""
This is a set of tools for numerically evaluating gradients of real-valued functions
Implements the formulas given near the top of http://cs231n.github.io/neural-networks-3/ for gradient checking
"""

import numpy as np

"""
-Given a numerical function f:R^2 -> R (taking a numpy array of real numbers and returning a real number) and a point x
 at which the function should be evaluated, returns the approximate gradient of all variables in x
 The given function should only take one argument, the numpy array. To evaluate parameterized functions, use
     numerical_gradient(lambda x: f(x, parameters...), x0, accuracy)
-The smaller the accuracy value, the more accurate the approximation will be
"""
def numerical_gradient(f, x, accuracy=1e-5):
    N, D = x.shape
    grads = np.zeros(x.shape)
    iterator = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not iterator.finished:
        index = iterator.multi_index
        original = x[index]
        x[index] += accuracy
        upper_bound = f(x)
        x[index] = original
        x[index] -= accuracy
        lower_bound = f(x)
        x[index] = original
        grads[index] = (upper_bound - lower_bound) / (2*accuracy)
        iterator.iternext()
    return grads
    
"""
-Given a numerical function (typically expected to be a "layer" inside a differentiable system) f:R^2 -> R^2, 
 a point x at which the function is evaluated, and the gradient flowing into the section of the system, returns
 the approximate gradient of all variables in x
 The given function should only take one argument, the numpy array. To evaluate parameterized functions, use
     numerical_gradient(lambda x: f(x, parameters...), x0, dout, accuracy)
-The smaller the accuracy value, the more accurate the approximation will be
"""
def numerical_gradient_layer(f, x, dout, accuracy=1e-5):
    grads = np.zeros(x.shape)
    iterator = np.nditer(x, flags = ['multi_index'], op_flags = ['readwrite'])
    while not iterator.finished:
        index = iterator.multi_index
        original = x[index]
        x[index] += accuracy
        upper_bound = f(x).copy() #copy: needed if function f alters its input, has no effect otherwise
        x[index] = original
        x[index] -= accuracy
        lower_bound = f(x).copy() #same as above
        x[index] = original
        grads[index] = np.sum((upper_bound - lower_bound) * dout) / (2*accuracy) #chain rule
        iterator.iternext()
    return grads
"""
-Used for determining the relative error of a numerical process
-Takes arrays giving the result of a process and the expected results
-Returns a numpy array expressing the relative error between corresponding result and expected values
"""

def relative_error(result, expected):
    return np.abs(result - expected) / np.maximum(np.abs(result) + np.abs(expected), 1e-8)
    