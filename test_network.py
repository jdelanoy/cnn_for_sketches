import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square

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
print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

transformer = caffe.io.Transformer({'data': solver.net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

path='./NYU_dataset/';

clusters = np.load(path+'clusters.npy');

#train the network and see intermediate results
for i in range(20):
    print 'step '+str(i)
    solver.step(1)
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
