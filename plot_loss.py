import numpy as np
import matplotlib.pyplot as plt
import h5py

train = np.load('/home/caffe_snapshot/train_loss_data_sketches.npy')
test = np.load('/home/caffe_snapshot/test_loss_data_sketches.npy')
#accu = np.load('/home/caffe_snapshot/accuracy_data_sketches.npy')

niter=train.shape[0]
#test = test[:4500/50]
test_iter=niter/test.shape[0]
train_iter=10

plt.plot(range(0,niter,train_iter),train[0:niter:train_iter])
plt.plot(range(0,niter,test_iter),test)
#plt.plot(range(0,niter,test_iter),accu)
plt.show()



