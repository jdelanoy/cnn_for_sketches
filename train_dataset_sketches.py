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

caffe.set_device(0)
caffe.set_mode_gpu()

#load network
solver = caffe.SGDSolver('caffe_network/solver_sketches.prototxt')
solver.net.copy_from('caffe_network/bvlc_reference_caffenet.caffemodel')
#solver.net.copy_from( '/home/caffe_snapshot/Trial1/snapshot_eigen_full_sketches_iter_5000.caffemodel')

# each output is (batch size, feature dim, spatial dim)
#print [(k, v.data.shape) for k, v in solver.net.blobs.items()]
#print [(k, v[0].data.shape) for k, v in solver.net.params.items()]

niter = 5000
test_interval=50
test_iter=10
train_loss = np.zeros(niter)
test_loss = np.zeros(niter/test_interval)
accuracy = np.zeros(niter/test_interval)

write_interval = 500
#train the network and see intermediate results
for i in range(niter):
    solver.step(1)
    train_loss[i] = solver.net.blobs['loss'].data
    
    if i % test_interval == 0:
        print 'Iteration', i, 'testing...'
        accu = 0
        loss = 0
        for test_it in range(test_iter):
            solver.test_nets[0].forward();
            loss += solver.test_nets[0].blobs['loss'].data
            accu +=solver.test_nets[0].blobs['accuracy'].data
        test_loss[i // test_interval] = loss/test_iter
        accuracy[i // test_interval] = accu/test_iter
        print train_loss[i]
        print test_loss[i // test_interval]

        if i % write_interval == 0:
            np.save('/home/caffe_snapshot/train_loss_data_sketches',train_loss[:i]);
            np.save('/home/caffe_snapshot/test_loss_data_sketches',test_loss[:i // test_interval]);
            np.save('/home/caffe_snapshot/accuracy_data_sketches',accuracy[:i // test_interval]);


np.save('/home/caffe_snapshot/train_loss_data_sketches',train_loss);
np.save('/home/caffe_snapshot/test_loss_data_sketches',test_loss);
np.save('/home/caffe_snapshot/accuracy_data_sketches',accuracy);

    
