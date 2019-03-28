import tensorflow as tf
import os
import importlib
import sys; sys.path += ['../compiler']
from compiler import *
import skimage.io

img_lo_path = ["/bigtemp/yy2bb/images/Images_320_240"]
img_med_path = ["/bigtemp/yy2bb/images/Images_640_480"]
img_hi_path = ["/bigtemp/yy2bb/images/Images_1280_960"]
batch_size = 10

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_ground_output(function, name, img_channel):

    image_reader = tf.WholeFileReader()
    all_filenames = []
    train_filename = name + '_train.tfrecords'
    test_filename = name + '_test.tfrecords'
    validate_filename = name + '_validate.tfrecords'
    
    for (paths, w, h) in [(img_lo_path, 320, 240), (img_med_path, 640, 480), (img_hi_path, 1280, 960)]:
        for path in paths:
        
            train_filename = '/bigtemp/yy2bb/' + name + '_' + str(w) + '_' + str(h) + '_train.tfrecords'
            test_filename = '/bigtemp/yy2bb/' + name + '_' + str(w) + '_' + str(h) + '_test.tfrecords'
            validate_filename = '/bigtemp/yy2bb/' + name + '_' + str(w) + '_' + str(h) + '_validate.tfrecords'
        
            _, filename_prefix = os.path.split(path)
            write_filenames = []
            new_path = path + '_ground_' + name
            if not os.path.isdir(new_path):
                os.mkdir(new_path)
            nfiles = len([file for file in os.listdir(path) if file.endswith('.jpg')])
            
            ntrain = nfiles // 10 * 8
            ntest = nfiles // 10
            nvalidate = ntest
            
            train_writer = tf.python_io.TFRecordWriter(train_filename)
            test_writer = tf.python_io.TFRecordWriter(test_filename)
            validate_writer = tf.python_io.TFRecordWriter(validate_filename)
            
            img_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + '/*.jpg'), shuffle=False)
            img_name, img_file = image_reader.read(img_queue)
            img = tf.cast(tf.image.decode_jpeg(img_file), tf.float32) / 256.0
            img.set_shape((w, h, img_channel))
            name_batch, img_batch = tf.train.batch([img_name, img], batch_size)
            output = function(img_batch)
            sess = tf.Session()
            tf.local_variables_initializer().run(session=sess)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)
            batch_num = nfiles // batch_size
            count = 0
            for i in range(batch_num):
                (filenames, out_imgs, in_imgs) = sess.run([name_batch, output, img_batch])
                out_imgs = out_imgs.astype(numpy.float32)
                in_imgs = in_imgs.astype(numpy.float32)
                for j in range(len(filenames)):
                    
                    feature = {'train/input': _bytes_feature(tf.compat.as_bytes(in_imgs[j, :, :, :].tostring())),
                               'train/output': _bytes_feature(tf.compat.as_bytes(out_imgs[j, :, :, :].tostring()))}
                
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    
                    if count < ntrain:
                        train_writer.write(example.SerializeToString())
                    elif count < ntrain + ntest:
                        test_writer.write(example.SerializeToString())
                    else:
                        validate_writer.write(example.SerializeToString())
                    count += 1

            train_writer.close()
            test_writer.close()
            validate_writer.close()
            coord.request_stop()
            coord.join(threads)
            
def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print('python get_ground_truth.py app_name img_channel gpu_name')
        print(' RGB only for now.')
        sys.exit(1)
        
    (app_name, img_channel, gpu_name) = args[:3]
    
    input_module = importlib.import_module(app_name)
    objective = input_module.objective
    if img_channel == '3':
        X = ImageRGB('x')
    elif img_channel == '1':
        X = ImageGray('x')
    else:
        raise 'Unknown img_channel'
    c = CompilerParams(verbose=0, allow_g=False, check_save=False, sanity_check=False)
    
    (_, output_module) = get_module_prefix(objective(X), c)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name
    return get_ground_output(output_module.f, app_name, img_channel)
    
if __name__ == '__main__':
    main()