import numpy as np
import h5py
from dataset_tools import *
from skimage import transform
import matplotlib.pyplot as plt
import time

path='/home/NYU_dataset/';

norm_size = [22.0, 29.0];
input_size=[228.0,304.0];
#####build train dataset with data augmentation

dataset = h5py.File(path+'train_dataset.h5', 'r')
images = dataset['images'];
normals =  dataset['normals'];
clusters =  dataset['clusters'][:,:];


nimg_input = images.shape[0]
batch_size = 2000;
nimg_output = 10*batch_size;


#initialize arrays
out_normals = np.ndarray((nimg_output, norm_size[0], norm_size[1]), np.uint8);
out_normals_center = np.ndarray((nimg_output), np.uint8);
out_images = np.ndarray((nimg_output , 3, input_size[0], input_size[1]), np.uint8);

beg = time.time()
for i in range(nimg_output):
    im = i%nimg_input
    img = transform.rescale(images[im], 0.6);
    norm = transform.rescale(normals[im], 0.6);
    
    #apply transformation
    img,norm=random_crop(img, norm, input_size);
    img,norm=random_scaling(img, norm, input_size);
    img = random_color(img);
    
    out_images[i] = img.transpose(2,0,1);
    
    center = norm[input_size[0]/2, input_size[1]/2].reshape((1,1,3));
    label = cluster_normals(center*2-1, clusters);
    out_normals_center[i] = np.argmax(label)
    
    normal_r = transform.rescale(norm, norm_size[0]/norm.shape[0])
    normal_r = central_crop(normal_r, norm_size);
    label = cluster_normals(normal_r*2-1, clusters);
    out_normals[i] = np.argmax(label,2)
    
    if((i+1)%500 == 0):
        print i+1
        print time.time()-beg
    #plt.subplot(1,3,1)
    #plt.imshow(img)
    #plt.subplot(1,3,2)
    #plt.imshow(norm)
    #plt.subplot(1,3,3)
    #plt.imshow(clusters[np.argmax(label,2)]*0.5+0.5)
    #plt.show()

#write datasets : one for 2000 images
print 'Writing train dataset'
train_center_filename = path +'train_normal_center'
train_full_filename = path +'train_normal_full'
f_train = open(train_center_filename+'.txt', 'w')
f_test = open(train_full_filename+'.txt', 'w')
for i in range(nimg_output/batch_size):
    with h5py.File(train_center_filename+str(i)+'.h5', 'w') as f:
         f['data'] = out_images[i*2000:(i+1)*2000]
         f['label'] = out_normals_center[i*2000:(i+1)*2000]
         f['clusters'] = clusters
    f_train.write(train_center_filename+str(i)+'.h5' + '\n')
    
    with h5py.File(train_full_filename+str(i)+'.h5', 'w') as f:
        f['data'] = out_images[i*2000:(i+1)*2000]
        f['label'] = out_normals[i*2000:(i+1)*2000]
        f['clusters'] = clusters
    f_test.write(train_full_filename+str(i)+'.h5' + '\n')


#####build test dataset with ground truth

dataset = h5py.File(path+'test_dataset.h5', 'r')
images = dataset['images'];
normals =  dataset['normals'];

nimg_input = images.shape[0]
img_size=images.shape[1:3];

test_normals = np.ndarray((nimg_input, norm_size[0], norm_size[1]), np.uint8);
test_normals_center = np.ndarray((nimg_input), np.uint8);
test_images = np.ndarray((nimg_input , 3, input_size[0], input_size[1]), np.uint8);


for i in range(nimg_input):
    img = transform.rescale(images[i], input_size[0]/img_size[0]);
    img = central_crop(img, input_size);
    test_images[i] = img.transpose(2,0,1);

    norm = normals[i];
    center = norm[img_size[0]/2, img_size[1]/2].reshape((1,1,3));
    label = cluster_normals(center*2-1, clusters);
    test_normals_center[i] = np.argmax(label)

    normal_r = transform.rescale(norm, norm_size[0]/norm.shape[0])
    normal_r = central_crop(normal_r, norm_size);
    label = cluster_normals(normal_r*2-1, clusters);
    test_normals[i] = np.argmax(label,2)


#write datasets
print 'Writing test dataset'
test_center_filename = path +'test_normal_center.h5'
with h5py.File(test_center_filename, 'w') as f:
    f['data'] = test_images
    f['label'] = test_normals_center
    f['clusters'] = clusters
with open(path+'test_normal_center.txt', 'w') as f:
    f.write(test_center_filename + '\n')

test_full_filename = path +'test_normal_full.h5'
with h5py.File(test_full_filename, 'w') as f:
    f['data'] = test_images
    f['label'] = test_normals
    f['clusters'] = clusters
with open(path+'test_normal_full.txt', 'w') as f:
    f.write(test_full_filename + '\n')
    
    
