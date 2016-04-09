import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py
import mpl_toolkits.axes_grid1 as axes_grid1
import matplotlib as mpl

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/caffe-future/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

path='/home/rendus/';
#snapshotpath = "/home/caffe_snapshot/T6/all_10e-4/"
#snapshotpath = "/home/caffe_snapshot/T1/vases+chairs_last_layers/"
snapshotpath = "/home/caffe_snapshot/"
#scale='coarse'
scale='scale1'
#scale='full'
niter=30000

caffe.set_mode_cpu()
net = caffe.Net('caffe_network/deploy_'+scale+'_sketches.prototxt',
                snapshotpath+'snapshot_eigen_sketches_'+scale+'_iter_'+str(niter)+'.caffemodel',
                caffe.TEST)

dset = 'candelabra'
trial='T6'
#dataset = h5py.File(path+trial+'_'+dset+'_train_dataset0.0.h5', 'r')
dataset = h5py.File(path+'modelnet_out/test/test_dataset0.h5', 'r')
images = dataset['data'];
normals = dataset['label'];
clusters = dataset['clusters'][:,:];

nb_test = 20
for it in range(nb_test):
    i = it * 1
    print i
    #load a test image
    net.blobs['data'].data[...] = images[i];
    out = net.forward();
    classif = out['output'][0];
    #classif = net.blobs['coarse'].data[0];
    classifMax = np.argmax(classif,0);
    classifRGB = clusters[classifMax]/2+0.5;
    confidence = np.array(classifMax, dtype=np.float32)
    for l in range(classifMax.shape[0]):
        for c in range(classifMax.shape[1]):
            prob = np.exp(classif[:,l,c])/np.sum(np.exp(classif[:,l,c]))
            confidence[l,c] = prob[classifMax[l,c]]
    #visualization of results
    plt.subplot(1, 4, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 4, 2)
    plt.imshow(clusters[normals[i].astype(int)]/2+0.5)
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
    plt.savefig('plots/modelnet/test_'+scale+'_BN_iter_'+str(niter)+'_'+str(it)+'.png', dpi=150, bbox_inches='tight')
    plt.show()

from skimage import io
for i in range(6):
    name = 'chairs'+str(i+1)+'_all'
    img = io.imread('chairs_sketches/'+name+'-01.png').transpose(2,0,1)[:3]
    net.blobs['data'].data[...] = img
    out = net.forward();
    classif = out['output'][0];
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
    plt.subplot(1, 3, 2)
    plt.imshow(classifRGB)
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(1, 3, 3)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.imshow(confidence, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.savefig('chairs_sketches/test_'+name+'.png', dpi=150, bbox_inches='tight')
    plt.show()
