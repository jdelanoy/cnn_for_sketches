from dataset_tools import *
from skimage import io
from skimage import transform
from skimage import filters, feature, morphology
from os import path
#import h5py
#import sklearn.cross_validation
import glob
import numpy as np
import sys
import matplotlib as plt

file = sys.argv[1]
#file = "_workspaces_modelnetDatabase_02691156_model_000003_3.png"
print "================================================="+file
#files = glob.glob('/home/rendus/modelnet/depth/*')



path_in = '/home/rendus/modelnet/test/'
path_out = '/home/rendus/modelnet/out/test/'

norm_size = [55.0, 74.0];
input_size=[228.0,304.0];
mid_size=[228.0*2,304.0*2];

#file=files[i]

dir_name, img_name = path.split(file)

#read and resize normal map
normal = io.imread(path_in+'normal/'+img_name)
#normal_cv = cv2.imread(path_in+'normal/'+img_name)
normal_r = transform.rescale(normal, max(mid_size[0]/normal.shape[0], mid_size[1]/normal.shape[1]))
normal_r = central_crop(normal_r, mid_size);
#read and resize depth map
depth = io.imread(path_in+'depth/'+img_name)
depth_r = transform.rescale(depth, max(mid_size[0]/depth.shape[0], mid_size[1]/depth.shape[1]))
depth_r = central_crop(depth_r, mid_size);

#compute line drawing
#normal_r = cv2.bilateralFilter(normal_cv, d=5, sigmaColor=100, sigmaSpace=100)

depth_cont =feature.canny(depth_r, sigma=1, low_threshold=0.05, high_threshold=0.1)
low_thresh=0.30
sigma=2
quant=False
norm0=feature.canny(normal_r[:,:,0], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm1=feature.canny(normal_r[:,:,1], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm2=feature.canny(normal_r[:,:,2], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm_cont1 = norm0+norm1+norm2

thresh_depth=depth_cont#> 0.05*np.max(depth_cont)
depth_cont2=morphology.dilation(thresh_depth.astype(int))
depth_cont3=morphology.dilation(morphology.dilation(depth_cont2))
diff = norm_cont1 - depth_cont3
cont = (diff>0.1) + thresh_depth
cont = 1-morphology.dilation(cont.astype(float))

gauss= filters.gaussian(normal_r, sigma=0, multichannel=True)
norm_cont2 = filters.sobel(gauss[:,:,1]) + filters.sobel(gauss[:,:,2]) + filters.sobel(gauss[:,:,0])
thresh_cont2 = norm_cont2 > 0.3*np.max(norm_cont2)

#plt.subplot(2,2,1)
#plt.imshow(normal_r)
#plt.subplot(2,2,2)
#plt.imshow(depth_r)
#plt.subplot(2,2,3)
#plt.imshow(norm_cont1, cmap='binary')
#plt.subplot(2,2,4)
#plt.imshow(cont, cmap='gray')
#plt.show()


cont = transform.rescale(cont, max(input_size[0]/cont.shape[0], input_size[1]/cont.shape[1]));
cont = central_crop(cont, input_size);
reshaped = cont.reshape((cont.shape[0],cont.shape[1],1))
img = np.concatenate((reshaped,reshaped, reshaped), axis=2)
io.imsave(path_out+'cont/'+img_name, img);

normal_r = transform.rescale(normal, max(norm_size[0]/normal.shape[0], norm_size[1]/normal.shape[1]))
normal_r = central_crop(normal_r, norm_size);
io.imsave(path_out+'normal/'+img_name, normal_r);

depth_r = transform.rescale(depth, max(norm_size[0]/depth.shape[0], norm_size[1]/depth.shape[1]))
depth_r = central_crop(depth_r, norm_size);
io.imsave(path_out+'depth/'+img_name, depth_r);


