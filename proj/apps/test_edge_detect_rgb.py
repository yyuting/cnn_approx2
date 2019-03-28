import sys; sys.path += ['../compiler']
from compiler import *
import scipy.ndimage
import numpy
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

kernel_sobel_x = numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def objective(X):
    sobel_x = ConstArrayExpr(kernel_sobel_x)
    sobel_y = array_transpose(sobel_x)
    
    X_gray = rgb2gray(X)
    gradient_x = conv2d(X_gray, sobel_x)
    gradient_y = conv2d(X_gray, sobel_y)
    mag = (gradient_x ** 2 + gradient_y ** 2) ** 0.5
    is_edge = mag >= 1
    return is_edge