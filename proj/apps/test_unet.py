"""
Code modified from tf_unet
https://github.com/jakeret/tf_unet
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy

in_channel = 3
out_channel = 3

x = tf.placeholder(tf.float32, shape=[None, None, None, in_channel], name='x')
y = tf.placeholder(tf.float32, shape=[None, None, None, out_channel], name='y')

nx = tf.shape(x)[1]
ny = tf.shape(x)[2]
x_images = tf.reshape(x, tf.stack([-1, nx, ny, in_channel]))

features = 4
filter_size = 3
pool_size = 2

stddev = numpy.sqrt(2 / (filter_size ** 2 * features))
w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, features], stddev=stddev))
w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, features, features], stddev=stddev))

b1 = tf.Variable(tf.constant(0.1, shape=[features]))
b2 = tf.Variable(tf.constant(0.1, shape=[features]))

#use SAME padding so we don't have to deal with size issues for now
conv1 = tf.nn.conv2d(x_images, w1, strides=[1, 1, 1, 1], padding="SAME")
# for simplicity, didn't include dropout for now
h_conv1 = tf.nn.relu(conv1 + b1)
conv2 = tf.nn.conv2d(h_conv1, w2, strides=[1, 1, 1, 1], padding="SAME")
down_h2 = tf.nn.relu(conv2 + b2)

pool = tf.nn.max_pool(down_h2, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding="SAME")
features *= 2

stddev = numpy.sqrt(2 / (filter_size ** 2 * features))
w3 = tf.Variable(tf.truncated_normal([filter_size, filter_size, features//2, features], stddev=stddev))
w4 = tf.Variable(tf.truncated_normal([filter_size, filter_size, features, features], stddev=stddev))

b3 = tf.Variable(tf.constant(0.1, shape=[features]))
b4 = tf.Variable(tf.constant(0.1, shape=[features]))

conv3 = tf.nn.conv2d(pool, w3, strides=[1, 1, 1, 1], padding="SAME")
h_conv3 = tf.nn.relu(conv3 + b3)
conv4 = tf.nn.conv2d(h_conv3, w4, strides=[1, 1, 1, 1], padding="SAME")
down_h4 = tf.nn.relu(conv4 + b4)

wd = tf.Variable(tf.truncated_normal([pool_size, pool_size, features//2, features], stddev=stddev))
bd = tf.Variable(tf.constant(0.1, shape=[features//2]))
down_shape = tf.shape(down_h4)
out_shape = tf.stack([down_shape[0], down_shape[1]*2, down_shape[2]*2, down_shape[3]//2])
h_deconv = tf.nn.relu(tf.nn.conv2d_transpose(down_h4, wd, out_shape, strides=[1, pool_size, pool_size, 1], padding="SAME") + bd)
h_deconv_concat = tf.concat([down_h2, h_deconv], 3)

w5 = tf.Variable(tf.truncated_normal([filter_size, filter_size, features, features//2], stddev=stddev))
w6 = tf.Variable(tf.truncated_normal([filter_size, filter_size, features//2, features//2], stddev=stddev))
b5 = tf.Variable(tf.constant(0.1, shape=[features//2]))
b6 = tf.Variable(tf.constant(0.1, shape=[features//2]))

conv5 = tf.nn.conv2d(h_deconv_concat, w5, strides=[1, 1, 1, 1], padding="SAME")
h_conv5 = tf.nn.relu(conv5 + b5)
conv6 = tf.nn.conv2d(h_conv5, w6, strides=[1, 1, 1, 1], padding="SAME")
up_h6 = tf.nn.relu(conv6 + b6)

weight = tf.Variable(tf.truncated_normal([1, 1, features//2, out_channel], stddev=stddev))
bias = tf.Variable(tf.constant(0.1, shape=[out_channel]))
conv = tf.nn.conv2d(up_h6, weight, strides=[1, 1, 1, 1], padding="SAME")
output_map = tf.nn.relu(conv + bias, name='output')

loss = tf.nn.l2_loss(output_map - y)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

train_x = numpy.load('train.npz')
test_x = numpy.load('test.npz')
train_y = numpy.load('train_out.npz')
test_y = numpy.load('test_out.npz')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

train_files = train_x.files
train_len = len(train_files)

saver = tf.train.Saver()

for i in range(1000):
    train_ind = train_files[numpy.random.randint(0, train_len)]
    sess.run(train_step, feed_dict={x: train_x[train_ind], y: train_y[train_ind]})
    #sess.run(train_step, feed_dict={x: train_x, y: train_y})
    if i % 100 == 0 and i != 0:
        saver.save(sess, '/localtmp/yuting/cnn_approx/saver2')
    print(loss.eval(feed_dict={x: train_x[train_ind], y: train_y[train_ind]}, session=sess))

out = []
for file in test_x.files:
    ans = output_map.eval(feed_dict={x: test_x[file], y: test_y[file]}, session=sess)
    out.append(ans)
numpy.savez('output7.npz', *out)
    
#ans = output_map.eval(feed_dict={x: test_x, y: test_y}, session=sess)
#numpy.save('/localtmp/yuting/cnn_approx/output6.npy', ans)