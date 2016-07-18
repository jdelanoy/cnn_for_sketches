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
import matplotlib as mpl
import matplotlib.pyplot as plt

#path_out = '/home/delanoy/database/CSG/out/'
#path_db = '/data/graphdeco/user/delanoy/CSG_database'
path_out = '/home/rendus/CSG/out_test/'
path_db = path_out
#deb=int(sys.argv[2])
#end=int(sys.argv[3])

id_batch = '0'#sys.argv[1]
i_batch=int(id_batch)
if i_batch>0:
    files = glob.glob(path_out+'/depth/'+id_batch+'[0-9][0-9][0-9]_*.png')
else:
    files = glob.glob(path_out+'/depth/???_*.png') + glob.glob(path_out+'/depth/??_*.png') + glob.glob(path_out+'/depth/?_*.png')


max_size = 2000


norm_size = [55.0, 74.0];
input_size=[228.0,304.0];

nimg = len(files);
if nimg < max_size:
    max_size = nimg

nb_max = ceil(float(nimg)/max_size)
print nimg
print nb_max
nb_clusters = 50;

normals_resize = np.ndarray((max_size, norm_size[0], norm_size[1], 4), np.float32);
depth_resize = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float32);
out_normals = np.ndarray((max_size, nb_clusters+1, norm_size[0], norm_size[1]), np.float16);
out_depth = np.ndarray((max_size, nb_clusters+1, norm_size[0], norm_size[1]), np.float16);
out_images = np.ndarray((max_size, 3, input_size[0], input_size[1]), np.float16);

for n in range(int(nb_max)):
#for n in range(deb,end+1):
    print 'batch '+str(n)
    #for i in range(max_size):
    for i in range(min(max_size, nimg-n*max_size)):
        im = n*max_size + i
        if im % 500 == 0:
            print im
        dir_name, img_name = path.split(files[im])

        if not path.exists(path_out+'normal/'+img_name) \
           or not path.exists(path_out+'depth/'+img_name) :
            print '-----------ERROR'
            #read and resize normal map
        else:
            out_images[i] = io.imread(path_out+'cont/'+img_name).transpose(2,0,1)/255.0;
            normals_resize[i] = io.imread(path_out+'normal/'+img_name)/255.0;
            depth_resize[i] = io.imread(path_out+'depth/'+img_name)/65535.0;

    print "All images loaded"
    if path.exists(path_out+'clusters_depth.npy'):
        clusters_depth =np.load(path_out+'clusters_depth.npy')
    else:
        print "Clustering depth"
        clusters_depth =np.ndarray((nb_clusters+1), np.float32);
        clusters_depth[:nb_clusters] = clustering_log_depth(depth_resize)
        clusters_depth[nb_clusters] = log(1)
        np.save(path_out+'clusters_depth', clusters_depth)
    #clusters_depth2 = np.exp(clusters_depth)
    #cluster on normals_resize
    if path.exists(path_out+'clusters.npy'):
        clusters_normal =np.load(path_out+'clusters.npy')
    else:
        print "Clustering normals"
        clusters_normal =np.ndarray((nb_clusters+1, 3), np.float32);
        clusters_normal[:nb_clusters] = clustering_k_means_array(normals_resize[:,:,:,:3]*2-1)
        clusters_normal[nb_clusters] = [0,0,0]
        np.save(path_out+'clusters', clusters_normal)

    for i in range(max_size):
        if i % 100 == 0:
            print i
        out_normals[i] = cluster_normals_gaussian(normals_resize[i,:,:,:3]*2-1, clusters_normal);
        out_depth[i] = cluster_depths_gaussian(depth_resize[i], clusters_depth);



    print 'Saving database'
    train_filename = path_db +'CSG_dataset'

    #f_train = open(train_filename+'.txt', 'w+')
    print 'Saving '+ train_filename+str(int(n+i_batch*4))+'.h5'
    with h5py.File(train_filename+str(int(n+i_batch*4))+'.h5', 'w') as f: #TODO
        f['data'] = out_images[:, :3, :, :]
        f['label_depth'] = out_depth
        f['label_normal'] = out_normals
        f['clusters_depth'] = clusters_depth
        f['clusters_normal'] = clusters_normal

    #f_train.close()
