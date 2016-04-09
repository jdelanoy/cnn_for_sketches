from dataset_tools import *
from skimage import io
from skimage import transform
from skimage import filters
from os import path
import h5py
#import sklearn.cross_validation
import glob
import sys
import numpy as np
from math import *

if sys.argc > 1:
    id_batch = sys.argv[1]
    i_batch=int(id_batch)
else:
    id_batch=""
    i_batch=0

max_size = 2000
#nb_trial = 6 
path_in = '/home/rendus/CSG/raw/'
path_out = '/home/rendus/CSG/out/'
#name = 'irons'
print path_out+'/depth/'+id_batch+'[0-9][0-9][0-9]_*.png'
files = glob.glob(path_out+'/depth/'+id_batch+'[0-9][0-9][0-9]_*.png')

norm_size = [55.0, 74.0];
input_size=[228.0,304.0];

nimg = len(files);
if nimg < max_size:
    max_size = nimg
nb_max = ceil(float(nimg)/max_size)
print nimg
nb_clusters = 50;

normals_resize = np.ndarray((max_size, norm_size[0], norm_size[1], 4), np.float32);
#depth_resize = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float32);
out_normals = np.ndarray((max_size, nb_clusters+1, norm_size[0], norm_size[1]), np.float16);
#out_depth = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float16);
out_images = np.ndarray((max_size, 3, input_size[0], input_size[1]), np.float16);

for n in range(int(nb_max)):
    #for i in range(max_size):
    for i in range(min(max_size, nimg-n*max_size)):
        im = n*max_size + i
        if im % 500 == 0:
            print begin+i
            dir_name, img_name = path.split(files[im])

    if not path.exists(path_out+'normal/'+img_name) \
       or not path.exists(path_out+'depth/'+img_name) :
        print '-----------TODO'
        #read and resize normal map
        normal = io.imread(path_in+'normal/'+img_name)
        normal_r = transform.rescale(normal, max(norm_size[0]/normal.shape[0], norm_size[1]/normal.shape[1]))
        normal_r = central_crop(normal_r, norm_size);
        normals_resize[i] = normal_r;

        #read and resize depth map
        #depth = io.imread(path_in+'depth/'+img_name)
        #depth_r = transform.rescale(depth, max(norm_size[0]/depth.shape[0], norm_size[1]/depth.shape[1]))
        #depth_r = central_crop(depth_r, norm_size);
        #depth_resize[i] = depth_r;
        #compute line drawing
        depth_cont = filters.sobel(depth)
        norm_cont = filters.sobel(normal[:,:,1]) + filters.sobel(normal[:,:,2]) + filters.sobel(normal[:,:,3])
        thresh_cont = norm_cont > 0.2*np.max(norm_cont)
        cont = thresh_cont+depth_cont < 0.05
        cont = transform.rescale(cont, max(input_size[0]/cont.shape[0], input_size[1]/cont.shape[1]));
        cont = central_crop(cont, input_size);
        reshaped = cont.reshape((228,304,1))
        img = np.concatenate((reshaped,reshaped, reshaped), axis=2)

        out_images[i] = img.transpose(2,0,1)*255.0;
        io.imsave(path_out+'cont/'+img_name, img);
        #io.imsave(path_out+'depth/'+img_name, depth_r);
        io.imsave(path_out+'normal/'+img_name, normal_r);
    else:
        out_images[i] = io.imread(path_out+'cont/'+img_name).transpose(2,0,1)/255.0;
        normals_resize[i] = io.imread(path_out+'normal/'+img_name)/255.0;
    #depth_resize[i] = io.imread(path_out+'depth/'+img_name)/255.0;


    #cluster on normals_resize
    if path.exists(path_out+'clusters.npy'):
        clusters_with0 =np.load(path_out+'clusters.npy')
    else:
        print "Clustering normals"
        clusters_with0 =np.ndarray((nb_clusters+1, 3), np.float32);
        clusters_with0[:nb_clusters] = clustering_k_means_array(normals_resize[:,:,:,:3])
        clusters_with0[nb_clusters] = [0,0,0]
        np.save(path_out+'clusters', clusters_with0)

    for i in range(max_size):
        out_normals[i] = cluster_normals_gaussian(normals_resize[i,:,:,:3]*2-1, clusters_with0);
        #out_normals[i] = np.argmax(label,2)

    print 'Saving database'
    train_filename = path_out +'test_dataset'

    #f_train = open(train_filename+'.txt', 'w+')
    #batch_size=max_size
    #nb_batch = ceil(float(max_size)/batch_size)
    beg = 0
    #for i in range(int(nb_batch)):
    with h5py.File(train_filename+str(int(n+i_batch*4))+'.h5', 'w') as f: #TODO
        f['data'] = out_images[beg:beg+batch_size, :3, :, :]
        f['label'] = out_normals[beg:beg+batch_size]
        f['clusters'] = clusters_with0
        beg += batch_size
        #f_train.write(train_filename+str(int(i+n*nb_batch))+'.h5' + '\n')

    #f_train.close()



