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

dataset = h5py.File(path+'train_normal_center.h5', 'r')
images = dataset['data'];
normals = dataset['label'];
gtnormals =  dataset['gt'];
clusters = np.load(path+'clusters.npy');

nb_img = images.shape[0];


niter = 10
test_interval=100
test_iter=350
train_loss = np.zeros(niter)
test_loss = np.zeros(niter)
input_size=[228.0,304.0];
#train the network and see intermediate results
for i in range(niter):
    #take image i
    im = i%nb_img;
    #plt.subplot(2,4,1)
    #plt.imshow(images[i])
    #plt.subplot(2,4,5)
    #plt.imshow(gtnormals[i])
    #apply random transform to image AND normals
    #crop
    img,norm=random_crop(images[im], gtnormals[im], input_size);
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
    normal_classif = cluster_normals(norm*2-1, clusters);
    label = normal_classif[input_size[0]/2, input_size[1]/2]
    solver.net.blobs['data'].data[:]=img;
    solver.net.blobs['label'].data[:] = label;
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data

    
    if i % test_interval == 0:
        print 'Iteration', i, 'testing...'
        loss = 0
        for test_it in range(test_iter):
            solver.test_nets[0].forward();
            loss += solver.test_nets[0].blobs['loss'].data
        test_loss[i // test_interval] = loss / test_iter
        print train_loss[i]
        print test_loss[i // test_interval]
    
    
    #plt.subplot(2, 5, 1)
    #plt.imshow(transformer.deprocess('data', solver.net.blobs['data'].data[0]))
    #plt.subplot(2, 5, 2)
    #plt.imshow(clusters[solver.net.blobs['label'].data[0].astype(int)])
    #print solver.net.blobs['label'].data[0]
    #plt.subplot(2, 5, 3)
    filters = solver.net.params['conv1'][0].data
    print 'conv1 param'
    print np.isnan(solver.net.params['conv1'][0].data).any()
    print np.isnan(solver.net.blobs['conv1'].data).any()
    #print filters[0,0]
    #vis_square(filters.transpose(0, 2, 3, 1))
    #plt.subplot(2, 5, 4)
    filters = solver.net.params['conv2'][0].data
    print 'conv2 param'
    print np.isnan(solver.net.params['conv2'][0].data).any()
    print np.isnan(solver.net.blobs['conv2'].data).any()
    #print filters[0,0]
    #vis_square(filters[:24,:24].reshape(24**2, 5, 5))
    #plt.subplot(2, 5, 5)
    filters = solver.net.params['conv3'][0].data
    print 'conv3 param'
    print np.isnan(solver.net.params['conv3'][0].data).any()
    print np.isnan(solver.net.blobs['conv3'].data).any()
    #print filters[0,0]
    #vis_square(filters[:24, :24].reshape(24**2, 3, 3))
    #plt.subplot(2, 5, 6)
    filters = solver.net.params['conv4'][0].data
    print 'conv4 param'
    print np.isnan(solver.net.params['conv4'][0].data).any()
    print np.isnan(solver.net.blobs['conv4'].data).any()
    #print filters[0,0]
    #vis_square(filters[:24, :24].reshape(24**2, 3, 3))
    #plt.subplot(2, 5, 7)
    filters = solver.net.params['conv5'][0].data
    print 'conv5 param'
    print np.isnan(solver.net.params['conv5'][0].data).any()
    print np.isnan(solver.net.blobs['conv5'].data).any()
    #print filters[0,0]
    feat = solver.net.blobs['conv5'].data[0, :100]
    #vis_square(feat, padval=1)
    #vis_square(filters[:24, :24].reshape(24**2, 3, 3))
    #plt.subplot(2, 5, 8)
    filters = solver.net.params['full1'][0].data
    print 'full1 param'
    print np.isnan(solver.net.params['full1'][0].data).any()
    print np.isnan(solver.net.blobs['f_1'].data).any()
    #print filters[:5, :5]
    #plt.imshow(filters[:100, :100])
    #plt.subplot(2, 5, 9)
    filters = solver.net.params['full2'][0].data
    print 'full2 param'
    print np.isnan(solver.net.params['full2'][0].data).any()
    print np.isnan(solver.net.blobs['f_2'].data).any()
    #print filters[:5, :5]
    #plt.imshow(filters[:100, :100])
    #plt.subplot(2, 5, 10)
    classif = solver.net.blobs['coarse'].data[0];
    print np.isnan(solver.net.blobs['coarse'].data).any()
    #plt.imshow(clusters[np.argmax(classif,0)])
    #print classif[:5,:5,:5]
    print '----------------------loss'
    print solver.net.blobs['loss'].data
    #plt.show()
