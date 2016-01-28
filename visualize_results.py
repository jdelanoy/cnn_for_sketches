import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/Downloads/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

path='/home/rendus/';


caffe.set_mode_cpu()
net = caffe.Net('caffe_network/deploy_coarse_sketches.prototxt',
                '/home/caffe_snapshot/snapshot_eigen_full_sketches_iter_5000.caffemodel',
                caffe.TEST)

dataset = h5py.File(path+'T5_test_dataset.h5', 'r')
images = dataset['data'];
normals = dataset['label'];
clusters = dataset['clusters'][:,:];

nb_test = 20
for i in range(nb_test):
    #load a test image
    net.blobs['data'].data[...] = images[i];
    out = net.forward();
    classif = out['coarse'][0];
    #print classif
    #print np.argmax(classif)
    #print normals[i]
    classifRGB = clusters[np.argmax(classif,0)];
    #visualization of results
    plt.subplot(1, 3, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0)/255.0)
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 3, 2)
    plt.imshow(clusters[normals[i]]/2+0.5)
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 3, 3)
    plt.imshow(classifRGB/2+0.5)
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('plots/T5/test_sketches_'+str(i)+'.png', dpi=150, bbox_inches='tight')
    plt.show()

