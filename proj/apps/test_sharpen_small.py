import sys; sys.path += ['../compiler']
from compiler import *
import scipy.ndimage
import numpy
import util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

kernel_sharp = numpy.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])

def objective(X):
    filter = ConstArrayExpr(kernel_sharp)
    return conv2d(X, filter)