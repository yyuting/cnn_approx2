import tensorflow as tf
import numpy

x = tf.placeholder(tf.float32, shape=(1, None, None, 1))
in_shape = tf.shape(x)
y = tf.image.resize_nearest_neighbor(x, (2*in_shape[1], 2*in_shape[2]))

conv = tf.nn.conv2d(y, numpy.empty([2, 2, 1, 1]), strides=[1, 1, 1, 1], padding="VALID")

x_val = numpy.zeros([1, 2, 3, 1])
x_val[0, 0, 0, 0] = 0
x_val[0, 1, 0, 0] = 1
x_val[0, 0, 1, 0] = 2
x_val[0, 1, 1, 0] = 3
x_val[0, 0, 2, 0] = 4
x_val[0, 1, 2, 0] = 5
feed_dict = {x: x_val}
sess = tf.Session()
y_val = sess.run(conv, feed_dict=feed_dict)
print(y_val.shape)