import importlib
import visualize_results_tools as visu
import h5py
#reload(visu)

import numpy as np
import matplotlib.pyplot as plt
from math import *
from skimage import io

# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/caffe-future/'
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.insert(0, '/user/delanoy/home/cnn_for_sketches/')

import caffe

path='/home/rendus/';
#snapshotpath = "/home/caffe_snapshot/T6/all_10e-4/"
#snapshotpath = "/home/caffe_snapshot/T1/vases+chairs_last_layers/"
snapshotpath = "/home/caffe_snapshot/cluster_pair/"
scale='full'
niter=5000

caffe.set_mode_cpu()
#net = caffe.Net('/user/delanoy/home/cnn_for_sketches/caffe_network/deploy_'+scale+'_gauss.prototxt',
#                snapshotpath+'snapshot_eigen_sketches_'+scale+'_iter_'+str(niter)+'.caffemodel',
net = caffe.Net('/user/delanoy/home/cnn_for_sketches/caffe_network/deploy_pair.prototxt',
                snapshotpath+'snapshot_pair_iter_'+str(niter)+'.caffemodel',
                caffe.TEST)

dataset = h5py.File(path+'CSG/out_test/CSG_dataset0.h5', 'r')
dataset = h5py.File('~/3D_depth/src/build/test_pair_dataset0.h5', 'r')
#dataset = h5py.File(path+'modelnet/out/test/modelnet_dataset0.h5', 'r')
#dataset = h5py.File(path+'modelnet/modelnet_databasemodelnet_dataset20.h5', 'r')
    
for i in range(10):
    save_path='plots/CSG/test_ortho_'+scale+'_iter_'+str(niter)+'_'+str(i)+'.png'
    print i
    visu.show_prediction_pair(dataset, net, i, \
                                 save=False, show=True, save_path=save_path);
           
save_path='/user/delanoy/home//3D_depth/src/build/proba_map_shapenet/'
for i in range(0,20):
    print i
    visu.compute_proba_tex(dataset, net, i,save=False, show=True)




for i in range(2):
    save_path='sketches/test_'+str(i+1)+'.png'
    img_name = 'sketches/original/'+str(i+1)+'.png'
    visu.show_prediction_sketch(dataset, net, img_name, \
                                save=False, show=True, save_path=save_path);

i=2
x=27
y=42

i=1
x=25
y=42


i=8
x=30
y=30

i=8
x=23
y=35



save_path='proba_distrib_'+str(i)+'_'+str(x)+'_'+str(y)+'.png'
visu.show_proba_distrib(dataset, net, i, x, y,save=False, show=True, save_path=save_path)


