import numpy as np
import matplotlib.pyplot as plt
from vis_square import vis_square
import h5py
from dataset_tools import *


# Make sure that caffe is on the python path:
caffe_root = '/user/delanoy/home/caffe/'
caffe_future_root = '/user/delanoy/home/caffe-future/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

#scale='coarse'
scale='full'
#scale='full'


#load network
solver = caffe.SGDSolver('caffe_network/solver_sketches_'+scale+'.prototxt')
#solver = caffe.AdamSolver('caffe_network/solver_sketches_adam.prototxt')
#solver.net.copy_from('caffe_network/bvlc_reference_caffenet.caffemodel')
solver.net.copy_from('/home/caffe_snapshot/full_CSG/snapshot_eigen_sketches_full_iter_5000.caffemodel')


niter = 1
test_interval=500
#test_iter=1000
train_loss_depth = np.zeros(niter/test_interval)
test_loss_depth = np.zeros(niter/test_interval)
accuracy_depth = np.zeros(niter/test_interval)
accuracy_depth2 = np.zeros(niter/test_interval)

train_loss_normal = np.zeros(niter/test_interval)
test_loss_normal = np.zeros(niter/test_interval)
accuracy_normal = np.zeros(niter/test_interval)
accuracy_normal2 = np.zeros(niter/test_interval)

write_interval = 1000
#train the network and see intermediate results
for i in range(niter):
    solver.step(1)
   
    if i % test_interval == 0:
        train_loss_depth[i // test_interval] = solver.net.blobs['loss_depth'].data
        test_loss_depth[i // test_interval] = solver.test_nets[0].blobs['loss_depth'].data
        accuracy_depth[i // test_interval] = solver.test_nets[0].blobs['accu_depth'].data
        accuracy_depth2[i // test_interval] = solver.test_nets[0].blobs['accu_depth2'].data
        train_loss_normal[i // test_interval] = solver.net.blobs['loss_normal'].data
        test_loss_normal[i // test_interval] = solver.test_nets[0].blobs['loss_normal'].data
        accuracy_normal[i // test_interval] = solver.test_nets[0].blobs['accu_normal'].data
        accuracy_normal2[i // test_interval] = solver.test_nets[0].blobs['accu_normal2'].data

        if i % write_interval == 0:
            np.save('/home/caffe_snapshot/train_loss_data_sketches_depth',train_loss_depth[:i // test_interval]);
            np.save('/home/caffe_snapshot/test_loss_data_sketches_depth',test_loss_depth[:i // test_interval]);
            np.save('/home/caffe_snapshot/accuracy_data_sketches_depth',accuracy_depth[:i // test_interval]);
            np.save('/home/caffe_snapshot/accuracy_data_sketches_depth2',accuracy_depth2[:i // test_interval]);
            np.save('/home/caffe_snapshot/train_loss_data_sketches_normal',train_loss_normal[:i // test_interval]);
            np.save('/home/caffe_snapshot/test_loss_data_sketches_normal',test_loss_normal[:i // test_interval]);
            np.save('/home/caffe_snapshot/accuracy_data_sketches_normal',accuracy_normal[:i // test_interval]);
            np.save('/home/caffe_snapshot/accuracy_data_sketches_normal2',accuracy_normal2[:i // test_interval]);


np.save('/home/caffe_snapshot/train_loss_data_sketches_depth',train_loss_depth);
np.save('/home/caffe_snapshot/test_loss_data_sketches_depth',test_loss_depth);
np.save('/home/caffe_snapshot/accuracy_data_sketches_depth',accuracy_depth);
np.save('/home/caffe_snapshot/accuracy_data_sketches_depth2',accuracy_depth2);

np.save('/home/caffe_snapshot/train_loss_data_sketches_normal',train_loss_normal);
np.save('/home/caffe_snapshot/test_loss_data_sketches_normal',test_loss_normal);
np.save('/home/caffe_snapshot/accuracy_data_sketches_normal',accuracy_normal);
np.save('/home/caffe_snapshot/accuracy_data_sketches_normal2',accuracy_normal2);
