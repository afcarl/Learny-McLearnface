import struct, os
import array as pyarray
import numpy as np

"""
Loads the MNIST dataset into numpy arrays.

The MNIST dataset may be downloaded at http://yann.lecun.com/exdb/mnist/
The dataset is parsed into numpy arrays, of size Nx768
where N is the number of images in the set, paired with a label array
of size Nx1. 

This assumes that the dataset has been downloaded and unpacked.
"""
def get_mnist(directory="."):
    training_images_filename = os.path.join(directory, "train-images.idx3-ubyte")
    training_labels_filename = os.path.join(directory, "train-labels.idx1-ubyte")
    test_images_filename = os.path.join(directory, "t10k-images.idx3-ubyte")
    test_labels_filename= os.path.join(directory, "t10k-labels.idx1-ubyte")
    training_images_file = open(training_images_filename, mode="rb")
    training_labels_file = open(training_labels_filename, mode="rb")
    test_images_file = open(test_images_filename, mode="rb")
    test_labels_file = open(test_labels_filename, mode="rb")
    training_images = parse_images_file(training_images_file)
    training_labels = parse_labels_file(training_labels_file)
    test_images = parse_images_file(test_images_file)
    test_labels = parse_labels_file(test_labels_file)
    return training_images, training_labels, test_images, test_labels
    
    
def parse_images_file(file):
    magic, num_images, num_rows, num_cols = struct.unpack(">IIII", file.read(16))
    data = pyarray.array("B", file.read())
    file.close()
    npdata = np.array(data)
    npdata = npdata.reshape((num_images, num_rows*num_cols))
    return npdata

def parse_labels_file(file):
    magic, num_labels = struct.unpack(">II", file.read(8))
    data = pyarray.array("b", file.read())
    file.close
    npdata = np.array(data)
    return npdata