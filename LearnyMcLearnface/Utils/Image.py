# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:26:55 2016

@author: Owner
"""

import numpy as np

"""
Takes a 2 or 3 dimensional numpy array
Assumes that dimensions 1 and 2 represent x and y pixel coordinates in the image, respectively
Assumes that if a third dimension is present, it represents channels of the image (rgb, etc.)
Pads the image with the indicated amount of zeroes along the first and second (spatial) dimensions only
"""
def zero_pad(X, pad):
    num_dims = len(X.shape)
    assert num_dims == 2 or num_dims == 3, 'The objected provided for zero-padding does not look like an image.'
    if num_dims == 2:
        return np.pad(X, pad, 'constant')
    if num_dims == 3:
        return np.pad(X, ((pad, pad), (pad, pad), (0, 0)), 'constant')