import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

import tensorflow as tf
import numpy
import json
import os
import sys; sys.path += ['../compiler']
import util
import skimage.io
import time
import scipy
import scipy.ndimage
import argparse_util

hi = 'hi'
med = 'med'
lo = 'lo'

def read_images_from_disk(input_queue):
    file_a = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3), tf.float32) / 256.0
    file_b = tf.decode_raw(tf.read_file(input_queue[1]), tf.float32)
    return file_a, file_b
    
def read_labeled_image_pairs(image_list_file, w, h, c, is_shuffle=True, bsize=10):
    f = open(image_list_file, 'r')
    img_a = []
    img_b = []
    for line in f:
        a, b = line[:-1].split(' ')
        img_a.append(a)
        img_b.append(b)
    a_tensor = tf.convert_to_tensor(img_a)
    b_tensor = tf.convert_to_tensor(img_b)
    input_queue = tf.train.slice_input_producer([a_tensor, b_tensor], shuffle=is_shuffle)
    img1, img2 = read_images_from_disk(input_queue)
    img2 = tf.reshape(img2, [w, h, c])
    img1.set_shape((w, h, c))
    img2.set_shape((w, h, c))
    img1_batch, img2_batch = tf.train.batch([img1, img2], bsize)
    return img1_batch, img2_batch
        
def get_default_data(prefix, name, sizes=[('lo', 10), ('med', 10), ('hi', 10)], is_shuffle=True):
    """
    return a list of data batches
    """
    ans = []
    for size in sizes:
        if size[0] == 'lo':
            w = 320
            h = 240
        elif size[0] == 'med':
            w = 640
            h = 480
        elif size[0] == 'hi':
            w = 1280
            h = 960
        else:
            raise
        filename = 'Images_' + str(w) + '_' + str(h) + '_bin' + name + prefix + '.txt'
        imgs, grounds = read_labeled_image_pairs(filename, w, h, 3, is_shuffle, size[1])
        ans.append((imgs, grounds, size[1]))
    return ans
    
def write_img(name, array):
    skimage.io.imsave(name, numpy.clip(array, 0.0, 1.0))
    
class ApproxNode:
    def __init__(self, 
                 cheating=False):
        self.in_pl = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.ground_pl = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        self.cheating = cheating
    
    def build_graph(self, nlayers=2,
                    nfeatures=3,
                    filter_size=3,
                    pool_size=2,
                    activation='none',
                    double_feature=True,
                    less_conv=False,
                    const_bias=False,
                    no_out_layer=False,
                    upsample='deconv',
                    padding='REFLECT'):
        """
        return: (output, weights)
        output: output tensor built upon the architecture given in arguments
        weights: dictionary containing all weights and biases tensors
        """
        if activation == 'none':
            activation_func = lambda x: x
        elif activation == 'relu':
            activation_func = tf.nn.relu
        else:
            raise
            
        weights = {}
        down_pyramid = []
        in_node = self.in_pl
        
        def get_std(f):
            return numpy.sqrt(2 / (filter_size ** 2 * f))
        
        def get_weight(shape, std, name):
            ans = tf.Variable(tf.truncated_normal(shape, stddev=std))
            weights[name+'w'] = ans
            return ans
            
        def get_bias(shape, std, name):
            if const_bias:
                ans = tf.Variable(tf.constant(0.0, shape=shape))
            else:
                ans = tf.Variable(tf.truncated_normal(shape, stddev=std))
            weights[name+'b'] = ans
            return ans
            
        def conv2d(input, filter, bias):
            if self.cheating:
                ans = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="VALID")
            else:
                if padding == 'REFLECT':
                    pad_x_size = int(filter.shape.as_list()[0] // 2)
                    pad_y_size = int(filter.shape.as_list()[0] // 2)
                    paddings = tf.constant([[0, 0], [pad_x_size, pad_x_size], [pad_y_size, pad_y_size], [0, 0]])
                    input_pad = tf.pad(input, paddings, "REFLECT")
                    ans = tf.nn.conv2d(input_pad, filter, strides=[1, 1, 1, 1], padding="VALID")
                elif padding == 'ZERO':
                    ans = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding="SAME")
                else:
                    raise
            return activation_func(ans+bias)
        
        size_diff = 0.0
        for layer in range(nlayers):
            if double_feature:
                features = 2 ** layer * nfeatures
                prev_features = features // 2
            else:
                features = nfeatures
                prev_features = nfeatures
                
            std = get_std(features)
            
            w1_name = 'down_conv' + str(layer) + '1'
            if layer == 0:
                w1 = get_weight([filter_size, filter_size, 3, features], std, w1_name)
            else:
                w1 = get_weight([filter_size, filter_size, prev_features, features], std, w1_name)
            b1 = get_bias([features], std, w1_name)
            conv1 = conv2d(in_node, w1, b1)
            
            size_diff += filter_size // 2
            
            w2_name = 'down_conv' + str(layer) + '2'
            w2 = get_weight([filter_size, filter_size, features, features], std, w2_name)
            b2 = get_bias([features], std, w2_name)
            conv2 = conv2d(conv1, w2, b2)
            size_diff += filter_size // 2
            
            down_pyramid.append(conv2)
            if layer < nlayers - 1:
                pool_down = tf.nn.avg_pool(conv1, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding="SAME")
                in_node = pool_down
                size_diff /= 2
            else:
                in_node = conv1
        
        for layer in range(nlayers-2, -1, -1):
            if double_feature:
                features = 2 ** (layer + 1) * nfeatures
                prev_features = features // 2
            else:
                features = nfeatures
                prev_features = nfeatures
            std = get_std(features)
            
            deconv_name = 'deconv' + str(layer)
            if upsample == 'deconv':
                w_deconv = get_weight([pool_size, pool_size, prev_features, features], std, deconv_name)
                b_deconv = get_bias([prev_features], std, deconv_name)
                in_shape = tf.shape(in_node)
                if double_feature:
                    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]//2])
                else:
                    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]])
                deconv = activation_func(tf.nn.conv2d_transpose(in_node, w_deconv, out_shape, strides=[1, pool_size, pool_size, 1], padding="SAME") + b_deconv)
            elif upsample == 'nn':
                # it's a simplification than in the u-net paper
                # because the learned weight would be hard to implement in tensorflow
                # and its effect should be learned in the following convolution
                in_shape = tf.shape(in_node)
                upsampled = tf.image.resize_nearest_neighbor(in_node, (pool_size*in_shape[1], pool_size*in_shape[2]))
                w_deconv = get_weight([1, 1, features, prev_features], std, deconv_name)
                b_deconv = get_bias([prev_features], std, deconv_name)
                deconv = conv2d(upsampled, w_deconv, b_deconv)
            else:
                raise
                
            size_diff *= 2
            
            pass_scalar = tf.Variable(tf.constant(1.0, shape=[]))
            weights['scalar'+str(layer)] = pass_scalar
            passthrough = pass_scalar * down_pyramid[layer] + deconv
            w1_name = 'up_conv' + str(layer) + '1'
            w1 = get_weight([filter_size, filter_size, prev_features, prev_features], std, w1_name)
            b1 = get_bias([prev_features], std, w1_name)
            conv1 = conv2d(passthrough, w1, b1)
            
            size_diff += filter_size // 2
            
            conv2 = conv1
            in_node = conv2
        
        output = in_node
        self.weights = weights
        self.output = output
        self.size_diff = size_diff
        
    def get_loss(self, loss_func, delta=0.3, scale=1.0):
        """
        loss_func: chooses which loss to apply
        """
        L1_loss = tf.abs(self.ground_pl - self.output)
        L2_loss = tf.square(self.ground_pl - self.output)
        
        if loss_func == 'L2':
            loss = L2_loss
        elif loss_func == 'L1':
           loss = L1_loss
        elif loss_func == 'L2_slope':
            small_res = 2.0 * delta * L1_loss
            large_res = L2_loss + delta ** 2
            condition = tf.less(L1_loss, delta)
            loss = tf.where(condition, small_res, large_ress)
        elif loss_func == 'L1+L2':
            loss = L2_loss + scale * L1_loss
        loss = tf.reduce_mean(loss)
        self.loss = loss
        
    def get_step(self, name='adam',
                 lrate=0.001):
        if name == 'adam':
            minimizer = tf.train.AdamOptimizer(learning_rate=lrate)
        elif name == 'sgd':
            minimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate)
        else:
            raise
        step = minimizer.minimize(self.loss)
        self.step = step
        
    def get_sess(self, read_from):
        if hasattr(self, 'sess'):
            sess = self.sess
        else:
            sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(self.weights)
        if read_from is not None:
            read_path = os.path.join(os.path.abspath(os.getcwd()), read_from, 'best')
            self.saver.restore(sess, tf.train.latest_checkpoint(read_path))
        self.sess = sess
        
    def start_queue(self):
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        
    def stop_queue(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        
    def get_feed_dict(self, in_batch, ground_batch):
        in_val, ground_val = self.sess.run([in_batch, ground_batch])
        feed_dict = {self.in_pl: in_val, self.ground_pl: ground_val}
        return feed_dict
        
    def inference(self, data_batches,
                  no_save=False,
                  acc_timing=False):
        self.start_queue()
        count = 0
        for data_batch in data_batches:
            input = data_batch[0]
            ground = data_batch[1]
            bsize = data_batch[2]
            feed_dict = self.get_feed_dict(input, ground)
            if acc_timing:
                self.sess.run([self.loss, self.output], feed_dict=feed_dict)
                T0 = time.time()
                for i in range(10):
                    self.sess.run([self.loss, self.output], feed_dict=feed_dict)
                T1 = time.time()
                print('time on this batch:', (T1 - T0)/10.0)
            if not no_save:
                loss_val, output_val = self.sess.run([self.loss, self.output], feed_dict=feed_dict)
                print('loss on this batch:', loss_val)
                input_val = feed_dict[self.in_pl]
                ground_val = feed_dict[self.ground_pl]
                for j in range(bsize):
                    write_img(str(count) + 'ground.png', ground_val[j, :, :, :])
                    write_img(str(count) + 'out.png', output_val[j, :, :, :])
                    write_img(str(count) + 'ainput.png', input_val[j, :, :, :])
                    count += 1
        self.stop_queue()
        
    def average_loss(self, data_batches):
        ans = 0.0
        scale = 0
        for data_batch in data_batches:
            input = data_batch[0]
            ground = data_batch[1]
            bsize = data_batch[2]
            feed_dict = self.get_feed_dict(input, ground)
            loss_val = self.sess.run(self.loss, feed_dict=feed_dict)
            ans += loss_val * bsize
            scale += bsize
        return ans / scale
        
    def train(self, data_batches, validate_batches,
              name='saver', iters=10000, tolerance=10, restarts=10,
              timing_only=False, batch_prefix=''):
        self.start_queue()
        best_loss = 1e100
        for k in range(restarts):
            print("restart:", k)
            loss_record = []
            best_this_run = 1e100
            to_best_count = 0
            is_best = False
            T0 = time.time()
            read_time = []
            back_prop_time = []
            for i in range(iters):
                for data_batch in data_batches:
                    T2 = time.time()
                    feed_dict = self.get_feed_dict(data_batch[0], data_batch[1])
                    T3 = time.time()
                    self.sess.run(self.step, feed_dict=feed_dict)
                    T4 = time.time()
                    read_time.append(T3 - T2)
                    back_prop_time.append(T4 - T3)
                if not timing_only:
                    if i % 50 == 0:
                        self.saver.save(self.sess, name + '/saver')
                        print(i)
                        validate_loss = self.average_loss(validate_batches)
                        if numpy.isnan(validate_loss):
                            is_best = False
                            break
                        loss_record.append([i, validate_loss])
                        print(validate_loss)
                        if validate_loss < best_loss:
                            best_loss = validate_loss
                            is_best = True
                            self.saver.save(self.sess, name + '/best/saver')
                        if validate_loss < best_this_run:
                            best_this_run = validate_loss
                            to_best_count = 0
                        else:
                            # Adam ocassionally increases loss drastically
                            if validate_loss > 100.0 * best_this_run:
                                break
                            to_best_count += 1
                            if to_best_count >= tolerance:
                                break
            T1 = time.time()
            print('time: ', T1 - T0)
            if not timing_only:
                loss_record = numpy.array(loss_record)
                numpy.save(name+'/loss'+str(k)+'.npy', loss_record)
            else:
                numpy.save(name+'/queue'+batch_prefix+'restart'+str(k)+'.npy', read_time)
                numpy.save(name+'/train'+batch_prefix+'restart'+str(k)+'.npy', back_prop_time)
            self.sess.run(tf.global_variables_initializer())
        self.stop_queue()
 
def plot_saved_loss(saver_name, xlim, ylim):
    all_files = os.listdir(saver_name)
    figure = pyplot.figure()
    for file in all_files:
        if file.endswith('.npy') and file.startswith('loss'):
            plot_name = file.replace('.npy', '').replace('loss', '')
            array = numpy.load(os.path.join(saver_name, file))
            pyplot.plot(array[:, 0], array[:, 1], label='run '+plot_name)
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.legend()
    figure.savefig('loss.png')
    
def plot_saved_time(saver_name, batch_prefix, nbatch, restarts, xlim, ylim):
    queue_all = numpy.empty(0)
    train_all = numpy.empty(0)
    figure = pyplot.figure()
    for i in range(restarts):
        queue_name = saver_name+'/queue'+batch_prefix+'restart'+str(i)+'.npy'
        train_name = saver_name+'/train'+batch_prefix+'restart'+str(i)+'.npy'
        queue_time = numpy.load(queue_name)
        train_time = numpy.load(train_name)
        queue_all = numpy.concatenate((queue_all, queue_time))
        train_all = numpy.concatenate((train_all, train_time))
    queue_all = numpy.reshape(queue_all, [len(queue_all)//nbatch, nbatch])
    train_all = numpy.reshape(train_all, [len(train_all)//nbatch, nbatch])
    for i in range(nbatch):
        pyplot.plot(numpy.arange(queue_all.shape[0]), queue_all[:, i], label='queue'+str(i))
        pyplot.plot(numpy.arange(train_all.shape[0]), train_all[:, i], label='train'+str(i))
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.legend()
    figure.savefig('time'+batch_prefix+'.png')
 
def main_args(is_test,
              data_batches,
              validate_batches,
              approx_kw={},
              nn_kw={},
              loss_func='L2',
              loss_kw={},
              minimizer_kw={},
              train_kw={},
              test_kw={},
              read_from=None):
              
    approx = ApproxNode(**approx_kw)
    approx.build_graph(**nn_kw)
    approx.get_loss(loss_func, **loss_kw)
    if not is_test:
        approx.get_step(**minimizer_kw)
    approx.get_sess(read_from)
    
    if is_test:
        approx.inference(data_batches, **test_kw)
    else:
        approx.train(data_batches, validate_batches, **train_kw)
 
def main(args):

    saver_name = args.saver_name
    is_test = args.is_test

    kw = {'approx_kw': {},
          'nn_kw': {},
          'loss_kw': {},
          'minimizer_kw': {},
          'train_kw': {'iters': args.iters,
                       'tolerance': args.tolerance,
                       'restarts': args.restarts},
          'test_kw': {'no_save': args.no_save,
                      'acc_timing': args.acc_timing}}
          
    if saver_name == 'saver144':
        kw['loss_func'] = 'L2'
        app_name = '_test_blur_large'
        kw['nn_kw']['nlayers'] = 5
        kw['nn_kw']['filter_size'] = 5
    elif saver_name == 'saver146':
        kw['loss_func'] = 'L2'
        app_name = '_test_blur_large'
        kw['nn_kw']['nlayers'] = 5
        kw['nn_kw']['filter_size'] = 5
        kw['nn_kw']['double_feature'] = False
    elif saver_name == 'saver148':
        kw['loss_func'] = 'L2'
        app_name = '_test_blur_large'
        kw['nn_kw']['nlayers'] = 5
        kw['nn_kw']['filter_size'] = 5
        kw['read_from'] = 'saver144'
        kw['minimizer_kw']['lrate'] = 0.0001
    elif saver_name == 'saver149':
        kw['loss_func'] = 'L2'
        app_name = '_test_blur_large'
        kw['nn_kw']['nlayers'] = 5
        kw['nn_kw']['filter_size'] = 5
        kw['read_from'] = 'saver144'
        kw['minimizer_kw']['lrate'] = 0.0001
    elif saver_name == 'saver150':
        kw['loss_func'] = 'L2'
        app_name = '_test_blur_large'
        kw['nn_kw']['nlayers'] = 5
        kw['nn_kw']['filter_size'] = 5
        
    if args.batch_sizes is not None:
        batch_sizes = eval(args.batch_sizes)
    else:
        batch_sizes = [('lo', 10), ('med', 10), ('hi', 10)]
    batch_prefix = '_'.join('_'.join(str(y) for y in x) for x in batch_sizes)
    
    if is_test:
        data_batches = get_default_data('test', app_name, sizes=batch_sizes)
        validate_batches = None
        kw['read_from'] = saver_name
        xlim = eval(args.xlim)
        ylim = eval(args.ylim)
        if args.plot_loss:
            plot_saved_loss(saver_name, xlim, ylim)
        if args.plot_time:
            plot_saved_time(saver_name, batch_prefix, len(batch_sizes), args.restarts, xlim, ylim)
    else:
        data_batches = get_default_data('train', app_name, sizes=batch_sizes)
        validate_batches = get_default_data('validate', app_name, sizes=batch_sizes)
        kw['train_kw']['name'] = saver_name
        kw['train_kw']['timing_only'] = args.train_timing
        kw['train_kw']['batch_prefix'] = batch_prefix
        
    main_args(is_test, data_batches, validate_batches, **kw)
 
if __name__ == '__main__':
    parser = argparse_util.ArgumentParser(description='Test on unet')
    parser.add_argument('saver_name', help='specify the saver directory')
    parser.add_argument('--is-test', dest='is_test', action='store_true', help='indicates testing')
    parser.add_argument('--is-train', dest='is_test', action='store_false', help='indicates training')
    parser.add_argument('--gpu-name', dest='gpu', default='0', help='name of GPU to use')
    parser.add_argument('--get-error', dest='error', action='store_true', help='get quantative error reports')
    parser.add_argument('--push-zero', dest='push_zero', type=float, default=1.0, help='in testing mode, cut weights in the same matrix to 0 if it is smaller than the maximum weight in that matrix times this ratio')
    parser.add_argument('--batch-sizes', dest='batch_sizes', default=None, help='determine batch sizes')
    parser.add_argument('--no-save', dest='no_save', action='store_true', help='run inference without saving output to images')
    parser.add_argument('--acc-timing', dest='acc_timing', action='store_true', help='run inference twice to get more accurate timing')
    parser.add_argument('--plot-loss', dest='plot_loss', action='store_true', help='plot saved loss and save to disk')
    parser.add_argument('--plot-time', dest='plot_time', action='store_true', help='plot time profiling')
    parser.add_argument('--xlim', dest='xlim', default='[0,10000]', help='x range in loss plot')
    parser.add_argument('--ylim', dest='ylim', default='[0.0,1.0]', help='y range in loss plot')
    parser.add_argument('--train-timing', dest='train_timing', action='store_true', help='a mode to time the training process with fixed iters')
    parser.add_argument('--iters', dest='iters', type=int, default=10000, help='number of iterations in training')
    parser.add_argument('--tolerance', dest='tolerance', type=int, default=10, help='determine when to end training')
    parser.add_argument('--restarts', dest='restarts', type=int, default=10, help='number of restarts in training')
    
    parser.set_defaults(is_test=True)
    parser.set_defaults(error=False)
    parser.set_defaults(no_save=False)
    parser.set_defaults(acc_timing=False)
    parser.set_defaults(plot_loss=False)
    parser.set_defaults(train_timing=False)
    parser.set_defaults(plot_time=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    main(args)