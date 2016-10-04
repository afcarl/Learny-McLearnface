# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:26:55 2016

@author: Owner
"""

import numpy as np

"""
An *image tensor* is a 4-dimensional tensor of floating point values (N, C, Y, X)
N indexes each image in the tensor
C indexes each channel of an image (RGB, HSV, Y'CbCr, etc.)
Y indexes pixels across the spatial height of a single channel of an image
X indexes pixels across the spatial width of a single channel of an image
"""

"""
Takes a 4 dimensional image tensor.
Pads each image with zeros the specified amount across all spacial dimensions (X and Y)
"""
def zero_pad(X, pad):
    num_dims = len(X.shape)
    assert num_dims == 4, 'The objected provided for zero-padding does not look like an image tensor.'
    return np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')

"""
Matlab's sliding im2col
This is a modified version of a very efficient implementation from Stanford's CS231N course
http://cs231n.github.io/
"""
def im2col(X, kernel_height, kernel_width, stride):
    N, C, H, W = X.shape
    assert (H - kernel_height) % stride == 0
    assert (W - kernel_width) % stride == 0
    steps_height = (H - kernel_height) // stride + 1
    steps_width = (W - kernel_width) // stride + 1
    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(steps_height), steps_width)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j0 = np.tile(np.arange(kernel_width), kernel_height * C)
    j1 = stride * np.tile(np.arange(steps_width), steps_height)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), kernel_height * kernel_width).reshape(-1, 1)
    cols = X[:, k, i, j]
    cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * C, -1)
    return cols