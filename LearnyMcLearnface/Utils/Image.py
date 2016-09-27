# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:26:55 2016

@author: Owner
"""

import numpy as np

"""
Takes a 4 dimensional image tensor.
The first coordinate indexes images in the tensor.
The second coordinate indexes channels for each image.
The third coordinate indexes the y-coordinate of pixels in each channel.
The fourth coordinate indexes the x-coordinate of pixels in each channel.
We only want to pad the images across their spatial dimensions, so we only pad across the last two dimensions.
"""
def zero_pad(X, pad):
    num_dims = len(X.shape)
    assert num_dims == 4, 'The objected provided for zero-padding does not look like an image tensor.'
    np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')