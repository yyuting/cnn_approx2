import tensorflow as tf
import os
import importlib
import sys; sys.path += ['../compiler']
from compiler import *
import skimage.io

img_lo_path = "/bigtemp/yy2bb/images/Images_320_240"
img_med_path = "/bigtemp/yy2bb/images/Images_640_480"
img_hi_path = "/bigtemp/yy2bb/images/Images_1280_960"
batch_size = 10

def get_ground_output(function, name, img_channel):

    image_reader = tf.WholeFileReader()
    all_filenames = []
    #for (path, w, h) in [(img_lo_path, 320, 240), (img_med_path, 640, 480), (img_hi_path, 1280, 960)]:
    for (path, w, h) in [(img_lo_path, 320, 240)]:
        write_filenames = []
        new_path = path + '_ground_' + name
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        nfiles = len([file for file in os.listdir(path) if file.endswith('.jpg')])
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
        for i in range(batch_num):
            (filenames, out_imgs) = sess.run([name_batch, output])
            for j in range(len(filenames)):
                full_img_path = filenames[j].decode("utf-8")
                _, filename = os.path.split(full_img_path)
                filename, _ = os.path.splitext(filename)
                write_img = numpy.squeeze(out_imgs[j, :, :, :])
                full_ground_path = os.path.join(new_path, filename+'.bin')
                #skimage.io.imsave(full_ground_path, numpy.clip(write_img, 0.0, 1.0))
                #numpy.save(full_ground_path, write_img.astype('f'))
                write_img.astype('f').tofile(full_ground_path)
                write_filenames.append(full_img_path + ' ' + full_ground_path)
        write_filenames.append('')
        #all_filenames.append(filename_prefix+'_bin_'+name+'.txt')
        #open(filename_prefix+'_bin_'+name+'.txt', 'w+').write('\n'.join(write_filenames))
        all_filenames.append(get_orig_name(path, name))
        open(all_filenames[-1], 'w+').write('\n'.join(write_filenames))
    
    for all_filename in all_filenames:
        split_files(all_filename)
        
def get_orig_name(path, name):
    _, prefix = os.path.split(path)
    return prefix + '_bin_' + name + '.txt'
    
def get_split_name(filename):
    orig_name, ext = os.path.splitext(filename)
    split_filenames = []
    for prefix in ['train', 'test', 'validate']:
        split_filenames.append(orig_name + prefix + ext)
    return tuple(split_filenames) 

def split_files(filename):
    lines = open(filename).read().split('\n')[:-1]
    nfiles = len(lines)
    #orig_name, ext = os.path.splitext(filename)
    ntrain = nfiles // 10 * 8
    ntest = nfiles // 10
    nvalidate = nfiles
    ans_train = '\n'.join(lines[:ntrain]) + '\n'
    ans_test = '\n'.join(lines[ntrain:ntrain+ntest]) + '\n'
    ans_validate = '\n'.join(lines[ntrain+ntest:]) + '\n'
    #name_train = orig_name + 'train'
    #name_test = orig_name + 'test'
    #name_validate = orig_name + 'validate'
    name_train, name_test, name_validate = get_split_name(filename)
    open(name_train, 'w+').write(ans_train)
    open(name_test, 'w+').write(ans_test)
    open(name_validate, 'w+').write(ans_validate)
            
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