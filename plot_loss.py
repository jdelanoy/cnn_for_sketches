import numpy as np
import matplotlib.pyplot as plt
import h5py

path='/home/caffe_snapshot/cluster_pair/'

train_depth = np.load(path+'train_loss_data_sketches_depth.npy')
test_depth = np.load(path+'test_loss_data_sketches_depth.npy')
accu_depth = np.load(path+'accuracy_data_sketches_depth.npy')
accu2_depth = np.load(path+'accuracy_data_sketches_depth2.npy')

train_normal = np.load(path+'train_loss_data_sketches_normal.npy')
test_normal = np.load(path+'test_loss_data_sketches_normal.npy')
accu_normal = np.load(path+'accuracy_data_sketches_normal.npy')
accu2_normal = np.load(path+'accuracy_data_sketches_normal2.npy')

niter=train_depth.shape[0]
#test = test[:4500/50]
test_iter=500#niter/test.shape[0]
inter=1
deb=1
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],train_depth[deb::inter], 'b')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],test_depth[deb::inter], 'b--')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],train_normal[deb::inter], 'r')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],test_normal[deb::inter], 'r--')
plt.legend(['Depth train loss','Depth test loss', 'Normal train loss','Normal test loss'], loc=0)
plt.show()


plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],accu_depth[deb::inter], 'b')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],accu2_depth[deb::inter], 'b--')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],accu_normal[deb::inter], 'r')
plt.plot(range(0,niter*test_iter,test_iter)[deb::inter],accu2_normal[deb::inter], 'r--')
plt.legend(['Depth accuracy, no background','Depth accuracy, background','Normal accuracy, no background','Normal accuracy, background'], loc=0)

plt.show()
