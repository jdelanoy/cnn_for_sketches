from dataset_tools import *
from skimage import io
from skimage import transform
from skimage import filters,feature,morphology
from os import path
#import h5py
#import sklearn.cross_validation
#import glob
import numpy as np
import sys
import matplotlib.pyplot as plt

file = sys.argv[1]

print "================================================="+file

path_in = '/home/rendus/CSG/raw_test/'
path_out = '/home/rendus/CSG/out_test/'

norm_size = [55.0, 74.0];
input_size=[228.0,304.0];
mid_size=[228.0*2,304.0*2];

dir_name, img_name = path.split(file)

#read and resize normal map
normal = io.imread(path_in+'normal/'+img_name)/255.0
normal_r = normal
#normal_r = transform.rescale(normal, max(mid_size[0]/normal.shape[0], mid_size[1]/normal.shape[1]), order=0)
#normal_r = central_crop(normal_r, mid_size);
#read and resize depth map
depth = io.imread(path_in+'depth/'+img_name)/255.0
depth_r = depth
#depth_r = transform.rescale(depth, max(mid_size[0]/depth.shape[0], mid_size[1]/depth.shape[1]), order=0)
#depth_r = central_crop(depth_r, mid_size);

#read and resize depth map
#t5 = io.imread(path_in+'T5/'+img_name)
#t5_r = transform.rescale(t5, max(norm_size[0]/t5.shape[0], norm_size[1]/t5.shape[1]))
#t5_r = central_crop(t5_r, norm_size);
#io.imsave(path_out+'T5/'+img_name, t5_r);

#compute line drawing
#depth_cont = filters.sobel(depth_r)
depth_cont =feature.canny(depth_r, sigma=1, low_threshold=0.05, high_threshold=0.1)

norm0=feature.canny(normal_r[:,:,0], sigma=4, low_threshold=0.25)
norm1=feature.canny(normal_r[:,:,1], sigma=4, low_threshold=0.25)
norm2=feature.canny(normal_r[:,:,2], sigma=4, low_threshold=0.25)
norm_cont1 = norm0+norm1+norm2

thresh_depth=depth_cont#> 0.05*np.max(depth_cont)
depth_cont2=morphology.dilation(thresh_depth.astype(int))
depth_cont3=morphology.dilation(morphology.dilation(depth_cont2))
diff = norm_cont1 - depth_cont3
cont = (diff>0.1) + thresh_depth
cont = 1-morphology.dilation(cont.astype(float))

#cont=(norm_cont+depth_cont) < 0.5

cont = transform.rescale(cont, max(input_size[0]/cont.shape[0], input_size[1]/cont.shape[1]));
cont = central_crop(cont, input_size);
reshaped = cont.reshape((cont.shape[0],cont.shape[1],1))
img = np.concatenate((reshaped,reshaped, reshaped), axis=2)
io.imsave(path_out+'cont/'+img_name, img);

normal_r = transform.rescale(normal, max(norm_size[0]/normal.shape[0], norm_size[1]/normal.shape[1]), order=0)
normal_r = central_crop(normal_r, norm_size);
io.imsave(path_out+'normal/'+img_name, normal_r);

depth_r = transform.rescale(depth, max(norm_size[0]/depth.shape[0], norm_size[1]/depth.shape[1]), order=0)
depth_r = central_crop(depth_r, norm_size);
io.imsave(path_out+'depth/'+img_name, depth_r);

