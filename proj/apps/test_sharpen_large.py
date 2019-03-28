import sys; sys.path += ['../compiler']
from compiler import *
import scipy.ndimage
import numpy
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_kernel_sharpening_large():
    kernel_size = 33
    kernel_half = kernel_size // 2
    sigma = 1.5
    kernel_base = numpy.zeros([kernel_size, kernel_size])
    kernel_base[kernel_half, kernel_half] = 1.0

    kernel_blur = scipy.ndimage.filters.gaussian_filter(kernel_base, sigma)

    filter_kernel_blur = scipy.ndimage.filters.gaussian_filter(kernel_blur, 1)
    alpha = 30
    sharpened_kernel = kernel_blur + alpha * (kernel_blur - filter_kernel_blur)
    return sharpened_kernel
    
def objective(X):
    kernel = create_kernel_sharpening_large()
    return conv2d(X, kernel)