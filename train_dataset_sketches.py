import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py
from dataset_tools import *


# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/caffe-master/'
caffe_future_root = '/user/delanoy/home/caffe-future/'
import sys
sys.path.insert(0, caffe_future_root + 'python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

scale='coarse'
#scale='scale1'
#scale='full'


#load network
solver = caffe.SGDSolver('caffe_network/solver_sketches_'+scale+'.prototxt')
#solver = caffe.AdamSolver('caffe_network/solver_sketches_adam.prototxt')
solver.net.copy_from('caffe_network/bvlc_reference_caffenet.caffemodel')
#solver.net.copy_from('/home/caffe_snapshot/coarse_modelnet/snapshot_eigen_sketches_coarse_iter_10000.caffemodel')


niter = 100000
test_interval=1000
test_iter=500
train_loss = np.zeros(niter/test_interval)
test_loss = np.zeros(niter/test_interval)
accuracy = np.zeros(niter/test_interval)

write_interval = 1000
#train the network and see intermediate results
for i in range(niter):
    solver.step(1)
   
    if i % test_interval == 0:
        train_loss[i // test_interval] = solver.net.blobs['loss'].data
        print 'Iteration', i, 'testing...'
        #accu = 0
        #loss = 0
        #for test_it in range(test_iter):
        #    solver.test_nets[0].forward();
        #    loss += solver.test_nets[0].blobs['loss'].data
        #    accu +=solver.test_nets[0].blobs['accuracy'].data
        test_loss[i // test_interval] = solver.test_nets[0].blobs['loss'].data#loss/test_iter
        #accuracy[i // test_interval] = solver.test_nets[0].blobs['accuracy'].data#accu/test_iter
        print train_loss[i // test_interval]
        print test_loss[i // test_interval]

        if i % write_interval == 0:
            np.save('/home/caffe_snapshot/train_loss_data_sketches',train_loss[:i // test_interval]);
            np.save('/home/caffe_snapshot/test_loss_data_sketches',test_loss[:i // test_interval]);
            #np.save('/home/caffe_snapshot/accuracy_data_sketches',accuracy[:i // test_interval]);


np.save('/home/caffe_snapshot/train_loss_data_sketches',train_loss);
np.save('/home/caffe_snapshot/test_loss_data_sketches',test_loss);
#np.save('/home/caffe_snapshot/accuracy_data_sketches',accuracy);
