import tensorflow as tf
import os
import sys; sys.path += ['../compiler']
from compiler import *
import get_ground_bin

def get_data_filenames(name):
    for path in [get_ground_bin.img_lo_path, 
                 get_ground_bin.img_med_path, 
                 get_ground_bin.img_hi_path]:
        orig_name = get_ground_bin.get_orig_name(path, name)
        split_names = get_ground_bin.get_split_name(orig_name)
        for split_name in split_names:
            if not os.path.exists(split_name):
                print('can not find dataset for approximation, run')
                print('python get_ground_bin app_name 3 gpu_name')
                print('to generate data')
                sys.exit(1)
        return split_names

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print('python approx_single.py app_name gpu_name')
        sys.exit(1)
    
    (app_name, gpu_name) = args[:3]
    
    data_filenames = get_data_filenames(app_name)
    
    input_module = importlib.import_module(app_name)
    approx_arch = input_module.approx_arch
    
    x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    y = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    approxnode = util.ApproxNode(x, y, 3, 3)
    ind_approx = approxnode.get_approx_ind('unet', **approx_arch)
    
    
    objective = input_module.objective
    
    X = ImageRGB('x')
    c = CompilerParams(verbose=0, allow_g=False, check_save=False, sanity_check=False)
    (_, output_module) = get_module_prefix(objective(X), c)
    
if __name__ == '__main__':
    main()