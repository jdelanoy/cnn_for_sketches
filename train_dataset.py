import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py
from dataset_tools import *

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/Downloads/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['image.cmap'] = 'gray'

caffe.set_device(0)
caffe.set_mode_gpu()

#load network
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('bvlc_reference_caffenet.caffemodel')

# each output is (batch size, feature dim, spatial dim)
#print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
#print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

path='./NYU_dataset/';

dataset = h5py.File(path+'train_normal.h5', 'r')
images = dataset['images'];
normals = dataset['label'];
gtnormals =  dataset['gt'];
clusters = np.load(path+'clusters.npy');

nb_img = images.shape[0];


niter = 5000
test_interval=50
test_iter=200
train_loss = np.zeros(niter)
test_loss = np.zeros(niter)
input_size=[228.0,304.0];
norm_size=[22.0,29.0];
#train the network and see intermediate results
for i in range(niter):
    #take image i
    im = i%nb_img;
    #img = central_crop(images[im].transpose(1,2,0), input_size);
    #plt.subplot(2,4,1)
    #plt.imshow(images[i].transpose(1,2,0))
    #plt.subplot(2,4,5)
    #plt.imshow(gtnormals[i].transpose(1,2,0))
    #apply random transform to image AND normals
    #crop
    img,norm=random_crop(images[im].transpose(1,2,0), gtnormals[im].transpose(1,2,0), input_size);
    #plt.subplot(2,4,2)
    #plt.imshow(img)
    #plt.subplot(2,4,6)
    #plt.imshow(norm)
    #random resize
    img,norm=random_scaling(img, norm, input_size);
    #plt.subplot(2,4,3)
    #plt.imshow(img)
    #plt.subplot(2,4,7)
    #plt.imshow(norm)
    #random color shift
    img = random_color(img);
    #plt.subplot(2,4,4)
    #plt.imshow(img)
    #plt.subplot(2,4,8)
    #plt.imshow(norm)
    #plt.show()
    #compute clustering
    #center = norm[input_size[0]/2, input_size[1]/2].reshape((1,1,3));
    #label = cluster_normals(center*2-1, clusters);
    normal_r = transform.rescale(norm, norm_size[0]/norm.shape[0])
    normal_r = central_crop(normal_r, norm_size);
    label = cluster_normals(normal_r*2-1, clusters);
    #label = normals[im];
    solver.net.blobs['data'].data[0]=img.transpose(2,0,1);
    #solver.net.blobs['label'].data[0] = np.argmax(label);
    solver.net.blobs['label'].data[0] = np.argmax(label,2);
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data
    
    if i % test_interval == 0:
        print 'Iteration', i, 'testing...'
        loss = 0
        for test_it in range(test_iter):
            solver.test_nets[0].forward();
            loss += solver.test_nets[0].blobs['loss'].data
        test_loss[i // test_interval] = loss/test_iter
        print train_loss[i]
        print test_loss[i // test_interval]

np.save('train_loss_data',train_loss);
np.save('test_loss_data',test_loss[:50]);

    
