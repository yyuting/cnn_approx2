import skimage
import skimage.io
import os
import numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#crop to the same size for easy batch processing
crop_size = True
high_half = (640, 480)
med_half = (320, 240)
low_half = (160, 120)

hi_full = (1280, 960)
med_full = (640, 480)
lo_full = (320, 240)
training_path = '/home/yy2bb/test_images/train'
testing_path = '/home/yy2bb/test_images/test'
#training_path = 'test_images/train'
#testing_path = 'test_images/test'

def read_and_save_images(path=None, save_as_image=False, w=320, h=240, files=None):
    if files is None:
        files = os.listdir(path)
    valid_imgs = []
    valid_files = []
    if save_as_image is True:
        main_path, sub_path = os.path.split(path)
        new_path = os.path.join(main_path, sub_path+'_'+str(w)+'_'+str(h))
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
    for file in files:
        try:
            img = skimage.io.imread(os.path.join(path, file))
            img = skimage.img_as_float(img)
            center_x = img.shape[0] // 2
            center_y = img.shape[1] // 2
            if center_x - w >= 0 and center_x + w <= img.shape[0] and center_y - h >= 0 and center_y + h <= img.shape[1] and img.shape[2] == 3:
                img = img[center_x-w:center_x+w, center_y-h:center_y+h, :]
                if save_as_image is True:
                    skimage.io.imsave(os.path.join(new_path, file), img)
                else:
                    valid_imgs.append(img)
                valid_files.append(file)
        except:
            pass
    if save_as_image is False:
        data = numpy.stack(valid_imgs, axis=0)
        print(data.shape)
        save_name = path.split('/')[-1]
        numpy.save(save_name + '.npy', data)
    return valid_files
    
def read_and_save_non_batch(path):
    files = os.listdir(path)
    imgs = []
    for file in files:
        try:
            img = skimage.io.imread(path + '/' + file)
            img = img[:img.shape[0]//2*2, :img.shape[1]//2*2, :]
            img = img.astype('f') / 256.0
            img = img.reshape((1,) + img.shape)
            if img.shape[3] == 3:
                imgs.append(img)
        except:
            pass
    save_name = path.split('/')[-1]
    numpy.savez(save_name + '.npz', *imgs)
    
#read_and_save_images(training_path)
#read_and_save_images(testing_path)
#read_and_save_non_batch(training_path)
#read_and_save_non_batch(testing_path)

"""
(w, h) = high_half
training_filenames = read_and_save_images(training_path, True, w, h)
testing_filenames = read_and_save_images(testing_path, True, w, h)
for (w, h) in [med_half, low_half]:
    read_and_save_images(training_path, True, w, h, training_filenames)
    read_and_save_images(testing_path, True, w, h, training_filenames)
"""
    
#read_and_save_images(training_path, True)
#read_and_save_images(testing_path, True)

def get_all_subdir_data(path):
    root_path, child_path = os.path.split(path)
    all_count = 0
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                all_count += 1
    
    per_size = all_count // 3
    lo_count = 0
    med_count = 0
    hi_count = 0
    lo_path = child_path + '_' + str(lo_full[0]) + '_' + str(lo_full[1])
    med_path = child_path + '_' + str(med_full[0]) + '_' + str(med_full[1])
    hi_path = child_path + '_' + str(hi_full[0]) + '_' + str(hi_full[1])
    
    if not os.path.isdir(os.path.join(root_path, lo_path)):
        os.mkdir(os.path.join(root_path, lo_path))
    if not os.path.isdir(os.path.join(root_path, med_path)):
        os.mkdir(os.path.join(root_path, med_path))
    if not os.path.isdir(os.path.join(root_path, hi_path)):
        os.mkdir(os.path.join(root_path, hi_path))
    
    for root, subdirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    img = skimage.io.imread(os.path.join(root, file))
                    img_t = numpy.transpose(img, axes=[1,0,2])
                    file_no_ext, _ = os.path.splitext(file)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        arrays =  [img, numpy.flip(numpy.flip(img, 0), 1),
                                   numpy.flip(img, 0), numpy.flip(img, 1),
                                   img_t, numpy.flip(numpy.flip(img_t, 0), 1),
                                   numpy.flip(img_t, 0), numpy.flip(img_t, 1)]
                        for i in range(len(arrays)):
                            array = skimage.img_as_float(arrays[i])
                            #if array.shape[0] >= hi_full[0] and array.shape[1] >= hi_full[1]:
                            #    array_reshape = array[:hi_full[0], :hi_full[1], :]
                            #    new_filename = os.path.join(root_path, hi_path, file_no_ext+str(i)+'.jpg')
                            #    skimage.io.imsave(new_filename, array_reshape)
                            #if array.shape[0] >= med_full[0] and array.shape[1] >= med_full[1]:
                            #    array_reshape = array[:med_full[0], :med_full[1], :] 
                            #    new_filename = os.path.join(root_path, med_path, file_no_ext+str(i)+'.jpg')
                            #    skimage.io.imsave(new_filename, array_reshape)
                            if array.shape[0] >= lo_full[0] and array.shape[1] >= lo_full[1]:
                                array_reshape = array[:lo_full[0], :lo_full[1], :]
                                new_filename = os.path.join(root_path, lo_path, file_no_ext+str(i)+'.jpg')
                                skimage.io.imsave(new_filename, array_reshape)
                    """
                        if hi_count < per_size:
                            if img.shape[0] >= hi_full[0] and img.shape[1] >= hi_full[1]:
                                img_reshape = skimage.img_as_float(img[:hi_full[0], :hi_full[1], :])
                                new_filename = os.path.join(root_path, hi_path, file)
                                skimage.io.imsave(new_filename, img_reshape)
                                hi_count += 1
                                continue
                            elif img.shape[1] >= hi_full[0] and img.shape[0] >= hi_full[1]:
                                img = numpy.transpose(img, axes=[1,0,2])
                                img_reshape = skimage.img_as_float(img[:hi_full[0], :hi_full[1], :])
                                new_filename = os.path.join(root_path, hi_path, file)
                                skimage.io.imsave(new_filename, img_reshape)
                                hi_count += 1
                                continue
                        if med_count < lo_count:
                            if img.shape[0] >= med_full[0] and img.shape[1] >= med_full[1]:
                                img_reshape = skimage.img_as_float(img[:med_full[0], :med_full[1], :])
                                new_filename = os.path.join(root_path, med_path, file)
                                skimage.io.imsave(new_filename, img_reshape)
                                med_count += 1
                                continue
                            elif img.shape[1] >= med_full[0] and img.shape[0] >= med_full[1]:
                                img = numpy.transpose(img, axes=[1,0,2])
                                img_reshape = skimage.img_as_float(img[:med_full[0], :med_full[1], :])
                                new_filename = os.path.join(root_path, med_path, file)
                                skimage.io.imsave(new_filename, img_reshape)
                                med_count += 1
                                continue
                        if  img.shape[0] >= lo_full[0] and img.shape[1] >= lo_full[1]:
                            img_reshape = skimage.img_as_float(img[:lo_full[0], :lo_full[1], :])
                            new_filename = os.path.join(root_path, lo_path, file)
                            skimage.io.imsave(new_filename, img_reshape)
                            lo_count += 1
                            continue
                        elif img.shape[1] >= lo_full[0] and img.shape[0] >= lo_full[1]:
                            img = numpy.transpose(img, axes=[1,0,2])
                            img_reshape = skimage.img_as_float(img[:lo_full[0], :lo_full[1], :])
                            new_filename = os.path.join(root_path, lo_path, file)
                            skimage.io.imsave(new_filename, img_reshape)
                            lo_count += 1
                            continue
                        #print('img of size', img.shape)    
                     """
                except:
                    pass
                    #print('img of size', img.shape)
                        
get_all_subdir_data('/home/yy2bb/test_images/Images')

def split_files(filename):
    lines = open(filename).read().split('\n')[:-1]
    nfiles = len(lines)
    orig_name, ext = os.path.splitext(filename)
    ntrain = nfiles // 10 * 8
    ntest = nfiles // 10
    nvalidate = nfiles
    ans_train = '\n'.join(lines[:ntrain]) + '\n'
    ans_test = '\n'.join(lines[ntrain:ntrain+ntest]) + '\n'
    ans_validate = '\n'.join(lines[ntrain+ntest:]) + '\n'
    name_train = orig_name + 'train'
    name_test = orig_name + 'test'
    name_validate = orig_name + 'validate'
    open(name_train+ext, 'w+').write(ans_train)
    open(name_test+ext, 'w+').write(ans_test)
    open(name_validate+ext, 'w+').write(ans_validate)
    
#split_files('Images_320_240_f2.txt')
#split_files('Images_640_480_f2.txt')
#split_files('Images_1280_960_f2.txt')