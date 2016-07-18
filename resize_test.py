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
#file='examples/cont/27910_4.png'
files=['0_1.png','0_6.png', '1_5.png','2_0.png','2_4.png', '2_5.png','3_1.png','3_3.png', '4_3.png','4_4.png','5_3.png', '6_3.png','6_6.png','6_7.png', '8_1.png','8_6.png','9_0.png', '9_6.png','10_0.png','10_2.png', '12_2.png','12_5.png','13_2.png', '13_5.png', '13_6.png']

l_thresh_rel=np.ndarray(len(files))
sigmas=np.ndarray(len(files))
l_thresh=np.ndarray(len(files))
BB=np.ndarray(len(files))
ratio=np.ndarray(len(files))
ccomplex=np.ndarray(len(files))


i=25

print "================================================="+file

path_in = '/home/rendus/CSG/raw_test/'
path_out = './examples_cont/'

norm_size = [55.0, 74.0];
input_size=[228.0,304.0];
mid_size=[228.0*2,304.0*2];

dir_name, img_name = path.split(files[i])
img_name='test_'+img_name
#read and resize normal map
normal = io.imread(path_in+'normal/'+img_name)
normal_r = transform.rescale(normal, max(mid_size[0]/normal.shape[0], mid_size[1]/normal.shape[1]))
normal_r = central_crop(normal_r, mid_size);
#read and resize depth map
depth = io.imread(path_in+'depth/'+img_name)
depth_r = transform.rescale(depth, max(mid_size[0]/depth.shape[0], mid_size[1]/depth.shape[1]))
depth_r = central_crop(depth_r, mid_size);

#read and resize depth map
#t5 = io.imread(path_in+'T5/'+img_name)
#t5_r = transform.rescale(t5, max(norm_size[0]/t5.shape[0], norm_size[1]/t5.shape[1]))
#t5_r = central_crop(t5_r, norm_size);
#io.imsave(path_out+'T5/'+img_name, t5_r);

BBOX = compute_BBOX(depth_r)
BB[i] = (BBOX[1][0]-BBOX[0][0])* (BBOX[1][1]-BBOX[0][1])*1.0/(mid_size[0]*mid_size[1])
ratio[i] = compute_ratio(depth_r)*1.0/(mid_size[0]*mid_size[1])
#compute line drawing
depth_cont =feature.canny(depth_r, sigma=1, low_threshold=0.05, high_threshold=0.1)
#depth_cont =filters.sobel(depth_r)

low_thresh=0.25
sigma=4
quant=False
norm0=feature.canny(normal_r[:,:,0], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm1=feature.canny(normal_r[:,:,1], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm2=feature.canny(normal_r[:,:,2], sigma=sigma, low_threshold=low_thresh, use_quantiles=quant)
norm_cont1 = norm0+norm1+norm2
sigmas[i]=sigma;
l_thresh[i]=low_thresh
#l_thresh_rel[i]=low_thresh

#cont = thresh_cont1+depth_cont
thresh_depth=depth_cont#> 0.05*np.max(depth_cont)
depth_cont2=morphology.dilation(thresh_depth.astype(int))
depth_cont3=morphology.dilation(morphology.dilation(depth_cont2))
diff = norm_cont1 - depth_cont3
cont = (diff>0.1) + thresh_depth
cont = 1-morphology.dilation(cont.astype(float))

gauss= filters.gaussian(normal_r, sigma=sigma, multichannel=True)
norm_cont2 = filters.sobel(gauss[:,:,1]) + filters.sobel(gauss[:,:,2]) + filters.sobel(gauss[:,:,0])
ccomplex[i]=np.sum(norm_cont2)

plt.subplot(2,3,1)
plt.imshow(depth_r, cmap='binary')
plt.subplot(2,3,4)
plt.imshow(normal_r, cmap='binary')
plt.subplot(2,3,2)
plt.imshow(gauss, cmap='binary')
plt.subplot(2,3,5)
plt.imshow(norm_cont2, cmap='binary')
plt.subplot(2,3,3)
plt.imshow(norm_cont1, cmap='binary')
plt.subplot(2,3,6)
plt.imshow(cont, cmap='gray')
plt.show()

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

plt.plot(BB[:11],sigmas[:11], 'ro')
plt.plot(BB[:11],l_thresh[:11], 'bo')
plt.show()

plt.plot(ratio[:11],sigmas[:11], 'ro')
plt.plot(ratio[:11],l_thresh[:11], 'bo')
plt.show()

def compute_BBOX(image):
    BBOX =[[image.shape[0], image.shape[1]], [0, 0]]
    for line in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[line,col] < 1.0:
                if line < BBOX[0][0]:
                    BBOX[0][0]= line
                if line > BBOX[1][0]:
                    BBOX[1][0] = line
                if col < BBOX[0][1]:
                    BBOX[0][1]= col
                if col > BBOX[1][1]:
                    BBOX[1][1] = col
    return BBOX;

def compute_ratio(image):
    nb=0
    for line in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[line,col] < 1.0:
                nb +=1;
    return nb;
