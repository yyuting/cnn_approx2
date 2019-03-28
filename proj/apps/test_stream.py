# Typical setup to include TensorFlow.
import tensorflow as tf
import skimage.io
import numpy

def test0():
    def read_labeled_image_pairs(image_list_file):
        f = open(image_list_file, 'r')
        img_a = []
        img_b = []
        for line in f:
            a, b = line[:-1].split(' ')
            img_a.append(a)
            img_b.append(b)
        return img_a, img_b

    def read_images_from_disk(input_queue):
        file_a = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3), tf.float32) / 256.0
        file_b = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[1]), channels=3), tf.float32)
        return file_a, file_b
        
    img_a, img_b = read_labeled_image_pairs('filenames.txt')
    a = tf.convert_to_tensor(img_a)
    b = tf.convert_to_tensor(img_b)
    
    input_queue = tf.train.slice_input_producer([a, b], shuffle=True)
    img1, img2 = read_images_from_disk(input_queue)
    img1.set_shape((640, 480, 3))
    img2.set_shape((640, 480, 3))
    
    img1_batch, img2_batch = tf.train.batch([img1, img2], 100)
    
    sess = tf.Session()
    # Required to get the filename matching to run.
    tf.local_variables_initializer().run(session=sess)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    x = tf.placeholder(tf.float32, (None, None, None, 3))
    y = tf.placeholder(tf.float32, (None, None, None, 3))
    loss = tf.reduce_sum(x - y)
    
    for i in range(10):
        print(sess.run(tf.reduce_mean(img1_batch)))
        print(sess.run(tf.reduce_mean(img2_batch)))
        print(sess.run(tf.reduce_sum((img1_batch - img2_batch) ** 2)))
        #img1_batch, img2_batch = tf.train.batch([img1, img2], 100)
        #loss_val = loss.eval(feed_dict={x: img1_batch, y: img2_batch}, session=sess)
        #print(loss_val)

    #image_tensor = sess.run(img1_batch - img2_batch)
    #print(image_tensor)
    
#test0()

def test1():
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(
        tf.train.match_filenames_once("/home/yy2bb/test_images/train_160_120/*.jpg"), shuffle=False)

    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    image_name, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.cast(tf.image.decode_jpeg(image_file), tf.float32) / 256.0
    
    image.set_shape((320, 240, 3))
    images, names = tf.train.batch([image, image_name], 10)

    # Start a new session to show example output.
    sess = tf.Session()
    # Required to get the filename matching to run.
    tf.local_variables_initializer().run(session=sess)

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Get an image tensor and print its value.
    img_vals, name_vals = sess.run([images, names])
    print(name_vals)
    _, name_vals = sess.run([images, names])
    print(name_vals)
    _, name_vals = sess.run([images, names])
    print(name_vals)
    images, names = tf.train.batch([image, image_name], 3)
    _, name_vals = sess.run([images, names])
    print(name_vals)
    _, name_vals = sess.run([images, names])
    print(name_vals)
    
    #image_tensor = sess.run(tf.cast(images, tf.float32) / 256.0)
    #print(image_tensor)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
    
def test2():
    f = open('train_160_120_npy_test_sharpen_largetrain.txt', 'r')
    
    #f = open('Images_320_240_npy_test_sharpen_largetrain.txt', 'r')
    img_a = []
    img_b = []
    for line in f:
        a, b = line[:-1].split(' ')
        img_a.append(a)
        img_b.append(b)
    a_tensor = tf.convert_to_tensor(img_a)
    b_tensor = tf.convert_to_tensor(img_b)
    input_queue = tf.train.slice_input_producer([a_tensor, b_tensor], shuffle=True)
    img1 = tf.cast(tf.image.decode_jpeg(tf.read_file(input_queue[0]), channels=3), tf.float32) / 256.0
    img2 = tf.decode_raw(tf.read_file(input_queue[1]), tf.float32)
    img1.set_shape((320, 240, 3))
    #img2.set_shape((320, 240, 3))
    img2 = tf.reshape(img2, [320, 240, 3])
    img2.set_shape((320, 240, 3))
    img1_batch, img2_batch = tf.train.batch([img1, img2], 10)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    img1_val, img2_val = sess.run([img1_batch, img2_batch])
    for i in range(10):
        skimage.io.imsave(str(i)+'ainput.png', numpy.clip(img1_val[i, :, :, :], 0.0, 1.0))
        skimage.io.imsave(str(i)+'ground.png', numpy.clip(img2_val[i, :, :, :], 0.0, 1.0))
    coord.request_stop()
    coord.join(threads)
test2()