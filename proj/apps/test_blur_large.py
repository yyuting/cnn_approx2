import sys; sys.path += ['../compiler']
from compiler import *
import scipy.ndimage
import numpy
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

kernel_size = 33
kernel_half = kernel_size // 2

def create_kernel():
    kernel_base = numpy.zeros([kernel_size, kernel_size])
    kernel_base[kernel_half, kernel_half] = 1.0
    kernel = scipy.ndimage.filters.gaussian_filter(kernel_base, 10.0)
    return kernel

def objective(X):
    kernel = create_kernel()
    return conv2d(X, kernel)