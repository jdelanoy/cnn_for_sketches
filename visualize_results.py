import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py
import mpl_toolkits.axes_grid1 as axes_grid1
import matplotlib as mpl

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/Downloads/caffe-master/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

path='/home/rendus/';
snapshotpath = "/home/caffe_snapshot/T6/all_10e-4/"
#snapshotpath = "/home/caffe_snapshot/T1/vases+chairs_last_layers/"
#snapshotpath = "/home/caffe_snapshot/"

caffe.set_mode_cpu()
net = caffe.Net('caffe_network/deploy_coarse_sketches.prototxt',
                snapshotpath+'snapshot_eigen_full_sketches_iter_2000.caffemodel',
                caffe.TEST)

dset = 'candelabra'
trial='T6'
dataset = h5py.File(path+trial+'_'+dset+'_train_dataset0.h5', 'r')
images = dataset['data'];
normals = dataset['label'];
clusters = dataset['clusters'][:,:];

nb_test = 10
for it in range(nb_test):
    i = it * 1
    print i
    #load a test image
    net.blobs['data'].data[...] = images[i];
    out = net.forward();
    classif = out['coarse'][0];
    #print classif
    #print np.argmax(classif)
    #print normals[i]
    classifMax = np.argmax(classif,0);
    classifRGB = clusters[classifMax]/2+0.5;
    confidence = np.array(classifMax, dtype=np.float32)
    for l in range(classifMax.shape[0]):
        for c in range(classifMax.shape[1]):
            prob = np.exp(classif[:,l,c])/np.sum(np.exp(classif[:,l,c]))
            confidence[l,c] = prob[classifMax[l,c]]
    #visualization of results
    plt.subplot(1, 4, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0)/255.0)
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 4, 2)
    plt.imshow(clusters[normals[i]]/2+0.5)
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 4, 3)
    plt.imshow(classifRGB)
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 4, 4)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.imshow(confidence, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('plots/'+trial+'/test_'+dset+'_'+str(it)+'_2K.png', dpi=150, bbox_inches='tight')
    plt.show()

from skimage import io
for i in range(0):
    name = 'chairs'+str(i+1)+'_all'
    img = io.imread('chairs_sketches/'+name+'-01.png').transpose(2,0,1)[:3]
    net.blobs['data'].data[...] = img[:,::-1]
    out = net.forward();
    classif = out['coarse'][0];
    #print classif
    #print np.argmax(classif)
    #print normals[i]
    classifRGB = clusters[np.argmax(classif,0)];
    #visualization of results
    plt.subplot(1, 2, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0)/255.0)
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 2, 2)
    plt.imshow(classifRGB/2+0.5)
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    #plt.savefig('chairs_sketches/test_'+name+'-2.png', dpi=150, bbox_inches='tight')
    plt.show()
