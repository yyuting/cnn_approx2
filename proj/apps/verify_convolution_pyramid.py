import tensorflow as tf
import numpy

sigma = 10.0
F = [4.8964, 0.0296, 0.2732, 1.5194, 2.5687, -0.1105, 0.0992, -0.0962, 8.9006, 0.0535, -0.0069, 0.0008]

h1 = numpy.array([[1, 4, 6, 4, 1]]) / 16.0
h1 = numpy.matmul(numpy.transpose(h1), h1)
h1_reshape = numpy.zeros([h1.shape[0], h1.shape[1], 3, 3])
h1_reshape[:, :, 0, 0] = h1[:, :]
h1_reshape[:, :, 1, 1] = h1[:, :]
h1_reshape[:, :, 2, 2] = h1[:, :]
h1 = tf.constant(h1_reshape, dtype=tf.float32)

h2 = F[0] * h1
h2_reshape = numpy.zeros([h2.shape[0], h2.shape[1], 3, 3])
h2_reshape[:, :, 0, 0] = h2[:, :]
h2_reshape[:, :, 1, 1] = h2[:, :]
h2_reshape[:, :, 2, 2] = h2[:, :]
h2 = tf.constant(h2_reshape, dtype=tf.float32)

g = numpy.array([[F[1], F[2], F[3], F[4], F[3], F[2], F[1]]])
g = numpy.matmul(numpy.transpose(g), g)
g_reshape = numpy.zeros([g.shape[0], g.shape[1], 3, 3])
g_reshape[:, :, 0, 0] = g[:, :]
g_reshape[:, :, 1, 1] = g[:, :]
g_reshape[:, :, 2, 2] = g[:, :]
g = tf.constant(g_reshape, dtype=tf.float32)

x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
in_node = x

for layer in range(7):
    a0 = in_node
    a = tf.
