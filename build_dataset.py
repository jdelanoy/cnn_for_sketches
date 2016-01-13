from dataset_tools import *
from skimage import io
from skimage import transform
import numpy as np
import h5py

import sklearn.cross_validation

#principal script

nb_img = 1449;
nb_clusters = 20;
img_size = [240.0, 320.0];
norm_size = [22.0, 29.0];
input_size=[228.0,304.0];

path='./NYU_dataset/';

#prepare files
normal_path=path + 'data/normal_';
depth_path=path + 'data/depth_';
image_path=path + 'data/image_';

#initialize arrays
normals = np.ndarray((nb_img, norm_size[0], norm_size[1]), np.uint8);
normalsRGB = np.ndarray((nb_img, 3, img_size[0], img_size[1]), np.uint8);
normals_center = np.ndarray((nb_img), np.uint8);
images = np.ndarray((nb_img, 3, img_size[0], img_size[1]), np.uint8);
crop_images = np.ndarray((nb_img, 3, 228, 304), np.uint8);
depths = np.ndarray((nb_img, img_size[0], img_size[1]), np.int32);

#load images
for i in range(nb_img):
    image_name = pad_string_with_0(i) + '.png';
    normals[i] = io.imread(normal_path+image_name);
    normalsRGB[i] = io.imread(normal_path+'RGB_'+image_name).transpose(2,0,1);
    normals_center[i] = normals[i,norm_size[0]/2.0,norm_size[1]/2.0];
    images[i] = io.imread(image_path+image_name).transpose(2,0,1);
    crop_images[i] = central_crop(io.imread(image_path+image_name),input_size).transpose(2,0,1);
    depths[i] = io.imread(depth_path+image_name);

#dataset with full normal maps
#separe train/test dataset
im, imT, cim, cimT, norm, normT, gt, gtT= sklearn.cross_validation.train_test_split(images, crop_images, normals, normalsRGB)

train_filename = path +'train_normal.h5'
test_filename = path +'test_normal.h5'

#write datasets
with h5py.File(train_filename, 'w') as f:
    f['data'] = cim
    f['images'] = im
    f['label'] = norm
    f['gt'] = gt
with open(path+'train_normal.txt', 'w') as f:
    f.write(train_filename + '\n')
with h5py.File(test_filename, 'w') as f:
    f['data'] = cimT
    f['images'] = imT
    f['label'] = normT
    f['gt'] = gtT
with open(path+'test_normal.txt', 'w') as f:
    f.write(test_filename + '\n')


#dataset with central pixel only
im, imT, cim, cimT, norm, normT, gt, gtT= sklearn.cross_validation.train_test_split(images, crop_images, normals_center, normalsRGB)

train_filename = path +'train_normal_center.h5'
test_filename = path +'test_normal_center.h5'

#write datasets
with h5py.File(train_filename, 'w') as f:
    f['data'] = cim
    f['images'] = im
    f['label'] = norm
    f['gt'] = gt
with open(path+'train_normal_center.txt', 'w') as f:
    f.write(train_filename + '\n')
#test dataset contains the ground truth normals
with h5py.File(test_filename, 'w') as f:
    f['data'] = cimT
    f['images'] = imT
    f['label'] = normT
    f['gt'] = gtT
with open(path+'test_normal_center.txt', 'w') as f:
    f.write(test_filename + '\n')

