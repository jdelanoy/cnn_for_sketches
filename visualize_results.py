import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/Downloads/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

path='./NYU_dataset/';


caffe.set_mode_cpu()
net = caffe.Net('deploy_coarse.prototxt',
                'snapshot_wang_iter_5000.caffemodel',
                caffe.TEST)

dataset = h5py.File(path+'test_normal.h5', 'r')
images = dataset['data'];
normals = dataset['label'];
gtnormals =  dataset['gt'];
clusters = np.load(path+'clusters.npy');

for i in range(images.shape[0]):
    #load a test image
    net.blobs['data'].data[...] = images[i];
    out = net.forward();
    classif = out['coarse'][0];
    classifRGB = clusters[np.argmax(classif,0)];
    #visualization of results
    plt.subplot(1, 4, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0)/255.0)
    plt.subplot(1, 4, 2)
    plt.imshow(gtnormals[i].transpose(1,2,0))
    plt.subplot(1, 4, 3)
    plt.imshow(clusters[normals[i]])
    plt.subplot(1, 4, 4)
    plt.imshow(classifRGB)
    plt.show()



    
