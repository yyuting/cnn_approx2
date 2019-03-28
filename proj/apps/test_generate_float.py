
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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def f(x):
    
    
    var000 = tf.constant(numpy.asarray(json.loads('[[[[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]]], [[[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]]], [[[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.07183506899653452, 0.0, 0.0], [0.0, 0.07183506899653452, 0.0], [0.0, 0.0, 0.07183506899653452]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]]], [[[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]]], [[[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]]]]')), dtype=tf.float32) # Expr, id: 139681943153072, Linenos for Expr: [263, 589, 731, 21, 84, 88], Linenos for codegen: [495, 451, 460, 490, 451, 138, 150, 176, 84, 88]
    var001_util_tensor_conv2d = util.tensor_conv2d(x,var000, filter_size=[5, 5]) # Expr, id: 139681943600488, Linenos for Expr: [263, 678, 671, 738, 21, 84, 88], Linenos for codegen: [495, 451, 138, 150, 176, 84, 88]
    return tf.cast(var001_util_tensor_conv2d, tf.float32)
    
def f2(x):
    """
    33x33 kernel
    """
    kernel_size = 33
    kernel_base = numpy.zeros([kernel_size, kernel_size])
    kernel_half = kernel_size // 2
    kernel_base[kernel_half, kernel_half] = 1.0
    kernel = scipy.ndimage.filters.gaussian_filter(kernel_base, 10.0)
    kernel_reshape = numpy.zeros([kernel_size, kernel_size, 3, 3])
    kernel_reshape[:, :, 0, 0] = kernel[:, :]
    kernel_reshape[:, :, 1, 1] = kernel[:, :]
    kernel_reshape[:, :, 2, 2] = kernel[:, :]
    var000 = tf.constant(kernel_reshape, tf.float32)
    #var000 = tf.constant(numpy.asarray(json.loads('[[[[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]]], [[[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]]], [[[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.07183506899653452, 0.0, 0.0], [0.0, 0.07183506899653452, 0.0], [0.0, 0.0, 0.07183506899653452]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]]], [[[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.05913986926416638, 0.0, 0.0], [0.0, 0.05913986926416638, 0.0], [0.0, 0.0, 0.05913986926416638]], [[0.04868825471235253, 0.0, 0.0], [0.0, 0.04868825471235253, 0.0], [0.0, 0.0, 0.04868825471235253]], [[0.032068886562107414, 0.0, 0.0], [0.0, 0.032068886562107414, 0.0], [0.0, 0.0, 0.032068886562107414]]], [[[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.038952921396240715, 0.0, 0.0], [0.0, 0.038952921396240715, 0.0], [0.0, 0.0, 0.038952921396240715]], [[0.03206888656210742, 0.0, 0.0], [0.0, 0.03206888656210742, 0.0], [0.0, 0.0, 0.03206888656210742]], [[0.02112241425389188, 0.0, 0.0], [0.0, 0.02112241425389188, 0.0], [0.0, 0.0, 0.02112241425389188]]]]')), dtype=tf.float32) # Expr, id: 139681943153072, Linenos for Expr: [263, 589, 731, 21, 84, 88], Linenos for codegen: [495, 451, 460, 490, 451, 138, 150, 176, 84, 88]
    var001_util_tensor_conv2d = util.tensor_conv2d(x,var000, filter_size=[kernel_size, kernel_size]) # Expr, id: 139681943600488, Linenos for Expr: [263, 678, 671, 738, 21, 84, 88], Linenos for codegen: [495, 451, 138, 150, 176, 84, 88]
    return tf.cast(var001_util_tensor_conv2d, tf.float32)
    
def sanity_check(test_cases=None):
    
    sess = tf.Session()
    util.check_output(test_cases, [x], _output_array, sess, check_save=False)
    return True
    
def read_images_from_disk(input_queue):
    file_a = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3), tf.float32) / 256.0
    file_b = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[1]), channels=3), tf.float32) / 256.0
    return file_a, file_b
    
batch_size = 10
    
def read_labeled_image_pairs(image_list_file, w, h, c, is_shuffle=True, bsize=batch_size):
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
    img1.set_shape((w, h, c))
    img2.set_shape((w, h, c))
    img1_batch, img2_batch = tf.train.batch([img1, img2], bsize)
    return img1_batch, img2_batch

def train_unet1():
    lo_filenames = "train_160_120.txt"
    med_filenames = "train_320_240.txt"
    hi_filenames = "train_640_480.txt"
    
    lo_imgs, lo_grounds = read_labeled_image_pairs(lo_filenames, 320, 240, 3)
    med_imgs, med_grounds = read_labeled_image_pairs(med_filenames, 640, 480, 3)
    hi_imgs, hi_grounds = read_labeled_image_pairs(hi_filenames, 1280, 960, 3)

    ind_unet = _approxnode.get_approx_ind('unet')
    
    def train():
        approx = _approxnode.approx_list[ind_unet]
        loss = approx['loss']
        temp = set(tf.all_variables())
        minimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = minimizer.minimize(loss)
        
        if 'sess' in approx.keys():
            sess = approx['sess']
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        else:
            sess = tf.Session()
            #tf.local_variables_initializer().run(session=sess)
            sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        
        T0 = time.time()
        for i in range(1000):
            for in_batch, ground_batch in [(lo_imgs, lo_grounds), (med_imgs, med_grounds), (hi_imgs, hi_grounds)]:
                in_val, ground_val = sess.run([in_batch, ground_batch])
                feed_dict = {_approxnode.in_pl: in_val, _approxnode.out_pl: ground_val}
                sess.run(train_step, feed_dict=feed_dict)
                print(loss.eval(feed_dict=feed_dict, session=sess))
        T1 = time.time()        
        approx['sess'] = sess
        coord.request_stop()
        coord.join(threads)
        return T1 - T0
    return train()
    
def train_unet2(read_from=None, is_shuffle=True, is_test=False, ntest=0,
                nfeatures=3, batch_sizes=[10, 10, 10], 
                loss_func='L2', is_crop=True, 
                is_const_bias=False, is_conv_only=False, 
                is_use_max=False, delta=0.05,
                nlayers=2, saver_dir='saver',
                functor_name='', restarts=10,
                scale=1.0, use_deconv=True,
                use_dropout=False, activation='none',
                filter_size=3, const_std=None,
                lrate=0.001, tolerance=10,
                report_error=False, double_feature=True,
                weight_regularizer=0, reg_bias=True):
    
    #functor_name = ''
    #functor_name = '_f2'
    
    #lo_filenames = "train_160_120.txt"
    #med_filenames = "train_320_240.txt"
    #hi_filenames = "train_640_480.txt"
    
    if not is_test:
        lo_filenames = functor_name + '_320_240_train.tfrecords'
        med_filenames = functor_name + '_640_480_train.tfrecords'
        hi_filenames = functor_name + '_1280_960_train.tfrecords'
    else:
        lo_filenames = functor_name + '_320_240_test.tfrecords'
        med_filenames = functor_name + '_640_480_test.tfrecords'
        hi_filenames = functor_name + '_1280_960_test.tfrecords'
    
    lo_validates = functor_name + '_320_240_validate.tfrecords'
    med_validates = functor_name + '_640_480_validate.tfrecords'
    hi_validates = functor_name + '_1280_960_validate.tfrecords'
    
    feature = {'train/input': tf.FixedLenFeature([], tf.string),
               'train/output': tf.FixedLenFeature([], tf.string)}
    #feature = {'train/input': tf.FixedLenFeature([], tf.float32),
    #           'train/output': tf.FixedLenFeature([], tf.float32)}
    
    def get_input_output(fname, w, h, bsize):
        queue = tf.train.string_input_producer([fname])
        """
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(queue)
        features = tf.parse_single_example(serialized_example, features=feature)
        in_img = tf.decode_raw(features['train/input'], tf.float32)
        out_img = tf.decode_raw(features['train/output'], tf.float32)
        in_img = tf.reshape(in_img, [w, h, 3])
        out_img = tf.reshape(out_img, [w, h, 3])
        
        if is_shuffle is True:
            in_imgs, out_imgs = tf.train.shuffle_batch([in_img, out_img], batch_size=bsize, capacity=2*bsize, min_after_dequeue=bsize)
        else:
            in_imgs, out_imgs = tf.train.batch([in_img, out_img], bsize)
        """
        
        step_size = 4 * bsize
        #reader = tf.TFRecordReader(options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB))
        reader = tf.TFRecordReader()
        _, queue_batch = reader.read_up_to(queue, step_size)
        features = tf.parse_example(queue_batch,features=feature)
        in_img = tf.decode_raw(features['train/input'], tf.float32)
        in_img = tf.reshape(in_img, [step_size, w, h, 3])
        out_img = tf.decode_raw(features['train/output'], tf.float32)
        out_img = tf.reshape(out_img, [step_size, w, h, 3])
        
        if is_shuffle:
            in_imgs, out_imgs = tf.train.shuffle_batch(
                                        [in_img, out_img],
                                        batch_size=bsize,
                                        capacity=4*bsize,
                                        min_after_dequeue=2*bsize,
                                        enqueue_many=True,
                                        allow_smaller_final_batch=False)
        else:
            in_imgs, out_imgs = tf.train.batch(
                                        [in_img, out_img],
                                        bsize,
                                        enqueue_many=True)
        
        return (in_imgs, out_imgs)
        
    lo_imgs, lo_grounds = get_input_output(lo_filenames, 320, 240, batch_sizes[0])
    med_imgs, med_grounds = get_input_output(med_filenames, 640, 480, batch_sizes[1])
    hi_imgs, hi_grounds = get_input_output(hi_filenames, 1280, 960, batch_sizes[2])
    
    lo_validate_imgs, lo_validate_grounds = get_input_output(lo_validates, 320, 240, batch_sizes[0])
    med_validate_imgs, med_validate_grounds = get_input_output(med_validates, 640, 480, batch_sizes[1])
    hi_validate_imgs, hi_validate_grounds = get_input_output(hi_validates, 1280, 960, batch_sizes[2])
    
    #new_approx = _approxnode.create_unet(feature_base=4, from_queue=True, in_batch=lo_imgs, out_batch=lo_grounds)
    if is_conv_only:
        nlayers = 1
        filter_size = 5
    #else:
    #    nlayers = 2
    #    filter_size = 3
    new_approx = _approxnode.create_unet(nlayers=nlayers, feature_base=nfeatures, filter_size=filter_size, from_queue=True, in_batch=lo_imgs, out_batch=lo_grounds, is_const_bias=is_const_bias, const_std=const_std, passthrough_op='plus', double_feature=double_feature)
    
    filter_half = filter_size // 2
    upper_left_corner = numpy.zeros([filter_half, filter_half])
    upper_side = numpy.zeros([filter_half])
    for i in range(filter_half):
        upper_left_corner[i, i] = ((i + filter_half + 1) / filter_size) ** 2.0
        for j in range(i):
            upper_left_corner[i, j] = (i + filter_half + 1) * (j + filter_half + 1) / (filter_size ** 2.0)
            upper_left_corner[j, i] = upper_left_corner[i, j]
        upper_side[i] = (i + filter_half + 1) / filter_size
    upper_right_corner = numpy.flip(upper_left_corner, 1)    
    lower_left_corner = numpy.flip(upper_left_corner, 0)
    lower_right_corner = numpy.flip(lower_left_corner, 1)
    
    def conv2d(input, filter, bias):
        pad_x_size = int(filter.shape.as_list()[0] // 2)
        pad_y_size = int(filter.shape.as_list()[0] // 2)
        paddings = tf.constant([[0, 0], [pad_x_size, pad_x_size], [pad_y_size, pad_y_size], [0, 0]])
        input_pad = tf.pad(input, paddings, "REFLECT")
        if activation == 'none':
            ans = tf.nn.conv2d(input_pad, filter, strides=[1, 1, 1, 1], padding="VALID") + bias
        elif activation == 'relu':
            ans = tf.nn.relu(tf.nn.conv2d(input_pad, filter, strides=[1, 1, 1, 1], padding="VALID") + bias)
        else:
            raise
        return ans
    
    def rebuild_graph():
        pool_size = new_approx['pool_size']
        feature_base = nfeatures
        losses = []
        outputs = []
        inputs = []
        for in_batch, ground_batch, w, h in [
        (lo_imgs, lo_grounds, 320, 240), 
        (med_imgs, med_grounds, 640, 480), 
        (hi_imgs, hi_grounds, 1280, 960),
        (lo_validate_imgs, lo_validate_grounds, 320, 240),
        (med_validate_imgs, med_validate_grounds, 640, 480),
        (hi_validate_imgs, hi_validate_grounds, 1280, 960)]:
            inputs.append(in_batch)
            down_pyramid = []
            in_node = in_batch
            size_diff = 0
            for layer in range(nlayers):
                features = 2 ** layer * feature_base
                w1 = new_approx['weights']['down_conv'+str(layer)+'1w']
                b1 = new_approx['biases']['down_conv'+str(layer)+'1b']
                w2 = new_approx['weights']['down_conv'+str(layer)+'2w']
                b2 = new_approx['biases']['down_conv'+str(layer)+'2b']
                conv1 = conv2d(in_node, w1, b1)
                conv2 = conv2d(conv1, w2, b2)
                down_pyramid.append(conv2)
                
                size_diff += 4 * (filter_size - 1)
                if layer < nlayers - 1:
                    pool_down = tf.nn.avg_pool(conv2, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1], padding="SAME")
                    in_node = pool_down
                    size_diff /= 2
                else:
                    in_node = conv2
                    
            for layer in range(nlayers-2, -1, -1):
                features = 2 ** (layer + 1) * feature_base
                w_deconv = new_approx['weights']['deconv'+str(layer)+'w']
                b_deconv = new_approx['biases']['deconv'+str(layer)+'b']
                in_shape = tf.shape(in_node)
                if double_feature:
                    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]//2])
                else:
                    out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]])
                if activation == 'none':
                    deconv = tf.nn.conv2d_transpose(in_node, w_deconv, out_shape, strides=[1, pool_size, pool_size, 1], padding="SAME") + b_deconv
                elif activation == 'relu':
                    deconv = tf.nn.relu(tf.nn.conv2d_transpose(in_node, w_deconv, out_shape, strides=[1, pool_size, pool_size, 1], padding="SAME") + b_deconv)
                else:
                    raise
                passthrough = down_pyramid[layer] + deconv
                
                w1 = new_approx['weights']['up_conv'+str(layer)+'1w']
                b1 = new_approx['biases']['up_conv'+str(layer)+'1b']
                w2 = new_approx['weights']['up_conv'+str(layer)+'2w']
                b2 = new_approx['biases']['up_conv'+str(layer)+'2b']
                conv1 = conv2d(passthrough, w1, b1)
                conv2 = conv2d(conv1, w2, b2)
                in_node = conv2
                size_diff *= 2
                size_diff += 4 * (filter_size - 1)
                
            w_out = new_approx['weights']['outputw']
            b_out = new_approx['biases']['outputb']
            conv_out = conv2d(in_node, w_out, b_out)
            
            if loss_func == 'L2':
                diff = tf.square(ground_batch - conv_out)
            elif loss_func == 'L1':
                diff = tf.abs(ground_batch - conv_out)
            elif loss_func == 'Huber':
                residual = tf.abs(ground_batch - conv_out)
                condition = tf.less(residual, delta)
                small_res = 0.5 * tf.square(residual)
                large_res = delta * residual - 0.5 * tf.square(delta)
                diff = tf.where(condition, small_res, large_res)
            elif loss_func == 'L2_down':
                residual = tf.abs(ground_batch - conv_out)
                small_res = tf.zeros(tf.shape(residual))
                large_res = tf.square(residual) - delta ** 2
                condition = tf.less(residual, delta)
                diff = tf.where(condition, small_res, large_res)
            elif loss_func == 'L2_slope':
                residual = tf.abs(ground_batch - conv_out)
                small_res = 2.0 * delta * residual
                large_res = tf.square(residual) + delta ** 2
                condition = tf.less(residual, delta)
                diff = tf.where(condition, small_res, large_res)
            elif loss_func == 'L1+L2':
                diff = tf.square(ground_batch - conv_out) + scale * tf.abs(ground_batch - conv_out)
            else:
                raise
            if is_crop:
                diff_crop = tf.slice(diff, [0, int(size_diff/2), int(size_diff/2), 0], [-1, w-int(size_diff), h-int(size_diff), -1])
                #diff_crop = tf.slice(diff, [0, 16, 16, 0], [-1, w-32, h-32, -1])
            else:
                diff_crop = diff
            if is_use_max:
                loss = tf.reduce_mean(diff_crop) + tf.reduce_max(diff_crop)
            else:
                loss = tf.reduce_mean(diff_crop)
            losses.append(loss)
            outputs.append(conv_out)
        return losses, outputs, inputs
    
    losses, outputs, inputs = rebuild_graph()
    
    if weight_regularizer > 0:
        loss_reg = 0
        if reg_bias:
            reg_vars = dict(new_approx['weights'], **new_approx['biases'])
        else:
            reg_vars = new_approx['weights']
        for weight in reg_vars.values():
            loss_reg += tf.reduce_sum(tf.square(weight))
        loss_reg *= weight_regularizer
        for i in range(len(losses)):
            losses[i] += loss_reg
            
    #lo_loss = new_approx['loss']
    lo_loss = losses[0]
    med_loss = losses[1]
    hi_loss = losses[2]
    #lo_output = new_approx['output']
    lo_output = outputs[0]
    med_output = outputs[1]
    hi_output = outputs[2]
    
    temp = set(tf.all_variables())
    minimizer = tf.train.AdamOptimizer(learning_rate=lrate)
    lo_train = minimizer.minimize(lo_loss)
    med_train = minimizer.minimize(med_loss)
    hi_train = minimizer.minimize(hi_loss)
    
    if 'sess' in new_approx.keys():
        sess = new_approx['sess']
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
    else:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
    saver = tf.train.Saver(dict(new_approx['weights'], **new_approx['biases']))
    if read_from is not None:
        read_path = os.path.join(os.path.abspath(os.getcwd()), read_from, 'best')
        #read_path = os.path.join(os.path.abspath(os.getcwd()), read_from)
        saver.restore(sess, tf.train.latest_checkpoint(read_path))
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    T0 = time.time()
    best_loss = 1e100
    if not is_test:
        for k in range(restarts):
            print("restart:", k)
            loss_record = []
            best_this_run = 1e100
            to_best_count = 0
            is_best = False
            for i in range(10000):
                for (loss, step) in [(lo_loss, lo_train), (med_loss, med_train), (hi_loss, hi_train)]:
                    #_, loss_val = sess.run([step, loss])
                    sess.run(step)
                if i % 50 == 0:
                    saver.save(sess, saver_dir + '/saver')
                    print(i)
                    validate_loss = numpy.mean(sess.run([losses[3], losses[4], losses[5]]))
                    if numpy.isnan(validate_loss):
                        is_best = False
                        break
                    loss_record.append([i, validate_loss])
                    print(validate_loss)
                    if validate_loss < best_loss:
                        best_loss = validate_loss
                        is_best = True
                        saver.save(sess, saver_dir + '/best/saver')
                    if validate_loss < best_this_run:
                        best_this_run = validate_loss
                        to_best_count = 0
                    else:
                        to_best_count += 1
                        if to_best_count >= tolerance:
                            break
            #saver.save(sess, saver_dir + '/saver')
            #if is_best:
            #    best_loss = validate_loss
            #    saver.save(sess, saver_dir + '/best/saver')
            loss_record = numpy.array(loss_record)
            numpy.save(saver_dir+'/loss'+str(k)+'.npy', loss_record)
            sess.run(tf.global_variables_initializer())
    else:
    
        if report_error:
            all_L2 = []
            all_psnr = []
            L2_sum = 0
            psnr_sum = 0
            file_sum = 0
            for (queue_filenames, grounds, unet_outputs, loss, input_imgs) in [(lo_filenames, lo_grounds, lo_output, lo_loss, inputs[0]), (med_filenames, med_grounds, med_output, med_loss, inputs[1]), (hi_filenames, hi_grounds, hi_output, hi_loss, inputs[2])]:
                #nfiles = len(open(queue_filenames).read().split('\n')) - 1
                nfiles = 0
                for record in tf.python_io.tf_record_iterator(queue_filenames):
                    nfiles += 1
                L2_list = []
                psnr_list = []
                for i in range(nfiles // 10):
                    ground_vals, unet_vals, loss_val, input_vals = sess.run([grounds, unet_outputs, loss, input_imgs])
                    for j in range(10):
                        output_ground = ground_vals[j, :, :, :]
                        output_img = unet_vals[j, :, :, :]
                        L2_error = numpy.mean((output_ground - output_img) ** 2)
                        psnr = 10.0 * numpy.log10(1.0 / L2_error)
                        L2_list.append(L2_error)
                        psnr_list.append(psnr)
                L2_list = numpy.asarray(L2_list)
                psnr_list = numpy.asarray(psnr_list)
                L2_sum += numpy.sum(L2_list)
                psnr_sum += numpy.sum(psnr_list)
                file_sum += nfiles
                all_L2.append(L2_list)
                all_psnr.append(psnr_list)
            print('L2: ', L2_sum / file_sum)
            print('psnr: ', psnr_sum / file_sum)
            numpy.savez('L2_err', all_L2[0], all_L2[1], all_L2[2])
            numpy.savez('psnr', all_psnr[0], all_psnr[1], all_psnr[2])
        else:          
            count = 0
            for i in range(ntest // batch_size):
                for (grounds, unet_outputs, loss, input_imgs) in [(lo_grounds, lo_output, lo_loss, inputs[0]), (med_grounds, med_output, med_loss, inputs[1]), (hi_grounds, hi_output, hi_loss, inputs[2])]:
                    ground_vals, unet_vals, loss_val, input_vals = sess.run([grounds, unet_outputs, loss, input_imgs])
                    print(loss_val)
                    for j in range(batch_size):
                        output_ground = numpy.squeeze(ground_vals[j, :, :, :])
                        output_img = numpy.squeeze(unet_vals[j, :, :, :])
                        input_img = numpy.squeeze(input_vals[j, :, :, :])
                        #skimage.io.imsave('test_output7/'+str(count)+'ground.png', numpy.clip(output_ground, 0.0, 1.0))
                        #skimage.io.imsave('test_output7/'+str(count)+'out1.png', numpy.clip(output_img, 0.0, 1.0))
                        #skimage.io.imsave('test_output7/'+str(count)+'ainput.png', numpy.clip(input_img, 0.0, 1.0))
                        skimage.io.imsave(str(count)+'ground.png', numpy.clip(output_ground, 0.0, 1.0))
                        skimage.io.imsave(str(count)+'out.png', numpy.clip(output_img, 0.0, 1.0))
                        skimage.io.imsave(str(count)+'ainput.png', numpy.clip(input_img, 0.0, 1.0))
                        count += 1
            
    T1 = time.time()     
    coord.request_stop()
    coord.join(threads)
    return T1 - T0

def get_ground_output():
    functor = f2
    functor_name = f2.__name__
    img_lo_path = ["/home/yy2bb/test_images/Images_320_240"]
    img_med_path = ["/home/yy2bb/test_images/Images_640_480"]
    img_hi_path = ["/home/yy2bb/test_images/Images_1280_960"]

    image_reader = tf.WholeFileReader()
    for (paths, w, h) in [(img_lo_path, 320, 240), (img_med_path, 640, 480), (img_hi_path, 1280, 960)]:
        for path in paths:
            _, filename_prefix = os.path.split(path)
            write_filenames = []
            new_path = path + '_ground_' + functor_name
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            nfiles = len([file for file in os.listdir(path) if file.endswith('.jpg')])
            img_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + '/*.jpg'), shuffle=False)
            img_name, img_file = image_reader.read(img_queue)
            img = tf.cast(tf.image.decode_jpeg(img_file), tf.float32) / 256.0
            img.set_shape((w, h, 3))
            name_batch, img_batch = tf.train.batch([img_name, img], batch_size)
            output = functor(img_batch)
            sess = tf.Session()
            tf.local_variables_initializer().run(session=sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            batch_num = nfiles // batch_size
            for i in range(batch_num):
                (filenames, out_imgs) = sess.run([name_batch, output])
                for j in range(len(filenames)):
                    full_img_path = filenames[j].decode("utf-8")
                    _, filename = os.path.split(full_img_path) 
                    write_img = numpy.squeeze(out_imgs[j, :, :, :])
                    full_ground_path = os.path.join(new_path, filename)
                    skimage.io.imsave(full_ground_path, numpy.clip(write_img, 0.0, 1.0))
                    write_filenames.append(full_img_path + ' ' + full_ground_path)
            """
            img_reshape = tf.expand_dims(img, axis=0)
            output = f(img_reshape)
            for i in range(nfiles % batch_size):
                (filename, out_img) = sess.run([img_name, output])
                full_img_path = filename.decode("utf-8")
                _, input_filename = os.path.split(full_img_path) 
                write_img = numpy.squeeze(out_img)
                full_ground_path = os.path.join(new_path, input_filename)
                skimage.io.imsave(full_ground_path, numpy.clip(write_img, 0.0, 1.0))
                write_filenames.append(full_img_path + ' ' + full_ground_path)
            """
            write_filenames.append('')
            open(filename_prefix+'_'+functor_name+'.txt', 'w+').write('\n'.join(write_filenames))
#get_ground_output()

if __name__ == '__main__':
#if False:
            
    x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    _out_x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    x._compiler_name = 'x'
    _output_array = f(x)
    _approxnode = util.get_approxnode(x,_out_x,_output_array)
    #dt1 = train_unet1()
    #dt2 = train_unet2(read_from='saver', is_test=True, ntest=10)
    #train_unet2()
    
    parser = argparse_util.ArgumentParser(description='Test on unet')
    parser.add_argument('saver_name', help='specify the saver directory')
    parser.add_argument('--is-test', dest='is_test', action='store_true', help='indicates testing')
    parser.add_argument('--is-train', dest='is_test', action='store_false', help='indicates training')
    parser.add_argument('--gpu-name', dest='gpu', default='0', help='name of GPU to use')
    parser.add_argument('--get-error', dest='error', action='store_true', help='get quantative error reports')
    
    parser.set_defaults(is_test=True)
    parser.set_defaults(error=False)
    args = parser.parse_args()
    
    saver_name = args.saver_name
    is_test = args.is_test
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    shuffle = True
    in_nfeatures = 3
    loss_crop = True
    const_bias = False
    conv_only = False
    use_max = False
    in_delta = 0.3
    in_nlayers = 2
    app_name = '_test_blur_approx'
    in_scale = 1.0
    in_deconv = True
    in_use_dropout = False
    in_activation = 'none'
    in_batch = [40, 20, 10]
    in_filtersize = 3
    in_const_std = None
    in_lrate = 0.001
    in_tolerance = 10
    in_double_feature = True
    in_weight_regularizer = 0
    in_reg_bias = True
    
    if saver_name == 'saver88':
        # failed
        in_loss = 'L1+L2'
        in_scale = 2.0
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_activation = 'relu'
        in_lrate = 0.01
    elif saver_name == 'saver89':
        # 6 in 10
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_activation = 'relu'
        in_lrate = 0.01
    elif saver_name == 'saver90':
        # 3 in 10, worse than 89
        in_loss = 'L2_slope'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_activation = 'relu'
        in_lrate = 0.01
    elif saver_name == 'saver91':
        # 1 in 3
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_lrate = 0.01
    elif saver_name == 'saver92':
        # worse than 91, 2 in 3
        in_loss = 'L2_slope'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_lrate = 0.01
    elif saver_name == 'saver93':
        # worse than 91, 2 in 2
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
    elif saver_name == 'saver94':
        # worse than 91, 2 in 2
        in_loss = 'L2_slope'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
    elif saver_name == 'saver97':
        # can't fully recover output, and can't drive weights to 0
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_weight_regularizer = 0.001
    elif saver_name == 'saver98':
        # none of the weights to 0
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_small'
        in_nlayers = 1
        in_weight_regularizer = 0.0001
    elif saver_name == 'saver95':
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_large'
        in_nlayers = 5
        in_filtersize = 5
    elif saver_name == 'saver96':
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_large'
        in_nlayers = 5
        in_filtersize = 5
        in_double_feature = False
    elif saver_name == 'saver99':
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_large'
        in_nlayers = 5
        in_filtersize = 5
        in_weight_regularizer = 0.1
    elif saver_name == 'saver100':
        in_loss = 'L2'
        loss_crop = False
        app_name = '_test_sharpen_large'
        in_nlayers = 5
        in_filtersize = 5
        in_weight_regularizer = 0.01
    else:
        raise
        
    if app_name == '_test_sharpen_small':
        app_name = 'test_sharpen_small'
    elif app_name == '_test_sharpen_large':
        app_name = '/bigtemp/yy2bb/test_sharpen_large'
    
    if args.error:
        train_unet2(is_shuffle=False, read_from=saver_name, 
                    is_test=True, ntest=10, 
                    nfeatures=in_nfeatures, batch_sizes=[10, 10, 10], 
                    loss_func=in_loss, is_crop=loss_crop, 
                    is_const_bias=const_bias, is_conv_only=conv_only,
                    is_use_max=use_max, delta=in_delta,
                    nlayers=in_nlayers, functor_name=app_name,
                    scale=in_scale, use_deconv=in_deconv,
                    use_dropout=in_use_dropout, activation=in_activation,
                    filter_size=in_filtersize, const_std=in_const_std,
                    lrate=in_lrate, tolerance=in_tolerance,
                    report_error=True, double_feature=in_double_feature,
                    weight_regularizer=in_weight_regularizer,
                    reg_bias=in_reg_bias)
    elif is_test:
        train_unet2(is_shuffle=shuffle, read_from=saver_name, 
                    is_test=True, ntest=10, 
                    nfeatures=in_nfeatures, batch_sizes=in_batch, 
                    loss_func=in_loss, is_crop=loss_crop, 
                    is_const_bias=const_bias, is_conv_only=conv_only,
                    is_use_max=use_max, delta=in_delta,
                    nlayers=in_nlayers, functor_name=app_name,
                    scale=in_scale, use_deconv=in_deconv,
                    use_dropout=in_use_dropout, activation=in_activation,
                    filter_size=in_filtersize, const_std=in_const_std,
                    lrate=in_lrate, tolerance=in_tolerance,
                    double_feature=in_double_feature,
                    weight_regularizer=in_weight_regularizer)
    else:
        train_unet2(is_shuffle=shuffle, saver_dir=saver_name, 
                    nfeatures=in_nfeatures, batch_sizes=in_batch, 
                    loss_func=in_loss, is_crop=loss_crop, 
                    is_const_bias=const_bias, is_conv_only=conv_only,
                    is_use_max=use_max, delta=in_delta,
                    nlayers=in_nlayers, functor_name=app_name,
                    scale=in_scale, use_deconv=in_deconv,
                    use_dropout=in_use_dropout, activation=in_activation,
                    filter_size=in_filtersize, const_std=in_const_std,
                    lrate=in_lrate, tolerance=in_tolerance,
                    double_feature=in_double_feature,
                    weight_regularizer=in_weight_regularizer,
                    reg_bias=in_reg_bias)
    #print(dt1)
    #print(dt2)