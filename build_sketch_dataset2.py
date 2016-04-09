from dataset_tools import *
from skimage import io
from skimage import transform
from os import path
import h5py
import sklearn.cross_validation
import glob
import numpy as np
from math import *

nb_trial = 6 
path_r = '/home/rendus/'
name = 'vases_large_'

files = glob.glob(path_r+'raw_data/*/depth/'+name+'*.png')

norm_size = [55.0, 74.0];
input_size=[228.0,304.0];

max_size = 2500

nimg = len(files);
if nimg < max_size:
    max_size = nimg
nb_max = ceil(float(nimg)/max_size)
print nimg
nb_clusters = 20;

normals_resize = np.ndarray((max_size, norm_size[0], norm_size[1], 4), np.float32);
depth_resize = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float32);

out_normals = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float16);
out_depth = np.ndarray((max_size, norm_size[0], norm_size[1]), np.float16);
out_images = np.ndarray((nb_trial, max_size, 4, input_size[0], input_size[1]), np.float16);

for n in range(int(nb_max)):
    print min(max_size, nimg-n*max_size)
    for i in range(min(max_size, nimg-n*max_size)):
        im = n*max_size + i
        if im % 100 == 0:
            print im
        dir_name, img_name = path.split(files[im])
        dir_name_raw = dir_name[:-5];
        dir_name_res = dir_name[:13]+'data_resized/'+dir_name[22:-5]

        if not path.exists(dir_name_res+'normal/'+img_name) \
           or not path.exists(dir_name_res+'depth/'+img_name) :
            print '-----------TODO'
            #read and resize line drawings
            #for t in range(nb_trial):
            #    img = io.imread(dir_name_raw+'T'+str(t+1)+'/'+img_name)[75:-75,208:-208]
            #    img = transform.rescale(img, max(input_size[0]/img.shape[0], input_size[1]/img.shape[1]));
            #    img = central_crop(img, input_size);
            #    out_images[t,i] = img.transpose(2,0,1)*255.0;
            #    io.imsave(dir_name_res+'T'+str(t+1)+'/'+img_name, img);
            for t in range(nb_trial):
                out_images[t,i] = io.imread(dir_name_res+'T'+str(t+1)+'/'+img_name).transpose(2,0,1)/255.0;
            #read and resize normal map
            normal = io.imread(dir_name_raw+'normal/'+img_name)[75:-75,208:-208]
            normal_r = transform.rescale(normal, max(norm_size[0]/normal.shape[0], norm_size[1]/normal.shape[1]))
            normal_r = central_crop(normal_r, norm_size);
            normals_resize[i] = normal_r;
            io.imsave(dir_name_res+'normal/'+img_name, normal_r);

            #read and resize depth map
            depth = io.imread(dir_name_raw+'depth/'+img_name)[75:-75,208:-208]
            depth_r = transform.rescale(depth, max(norm_size[0]/depth.shape[0], norm_size[1]/depth.shape[1]))
            depth_r = central_crop(depth_r, norm_size);
            depth_resize[i] = depth_r;
            io.imsave(dir_name_res+'depth/'+img_name, depth_r);

        else:
            for t in range(nb_trial):
                out_images[t,i] = io.imread(dir_name_res+'T'+str(t+1)+'/'+img_name).transpose(2,0,1)/255.0;
            normals_resize[i] = io.imread(dir_name_res+'normal/'+img_name)/255.0;
            depth_resize[i] = io.imread(dir_name_res+'depth/'+img_name)/255.0;


    #cluster on normals_resize
    if path.exists(path_r+'clusters.npy'):
        clusters_with0 =np.load(path_r+'clusters.npy')
    else:
        print "Clustering normals"
        clusters_with0 =np.ndarray((nb_clusters+1, 3), np.float32);
        clusters_with0[:nb_clusters] = clustering_k_means_array(normals_resize[:,:,:,:3])
        clusters_with0[nb_clusters] = [0,0,0]
        np.save(path_r+'clusters', clusters_with0)

    for i in range(max_size):
        label = cluster_normals(normals_resize[i,:,:,:3]*2-1, clusters_with0);
        out_normals[i] = np.argmax(label,2)

    #separe train/test dataset
    #600 first is  test set
    nb_test = 0
    for t in range(nb_trial):
        print 'Saving T'+str(t+1)
        train_filename = path_r +'T'+str(t+1)+'_'+name+'train_dataset'

        f_train = open(train_filename+'.txt', 'w+')
        batch_size=2500
        nb_batch = ceil(float(max_size-nb_test)/batch_size)
        beg = nb_test
        for i in range(int(nb_batch)):
            with h5py.File(train_filename+str(i+n*nb_batch)+'.h5', 'w') as f:
                 f['data'] = out_images[t, beg:beg+batch_size, :3, :, :]
                 f['label'] = out_normals[beg:beg+batch_size]
                 f['clusters'] = clusters_with0
            beg += batch_size
            f_train.write(train_filename+str(i+n*nb_batch)+'.h5' + '\n')

        f_train.close()

        if nb_test > 0:
            test_filename = path_r +'T'+str(t+1)+'_'+name+'test_dataset'

            f_test = open(test_filename+'.txt', 'w')
            batch_size=2500
            nb_batch = ceil(float(nb_test)/batch_size)
            beg = 0
            for i in range(int(nb_batch)):
                with h5py.File(test_filename+str(i)+'.h5', 'w') as f:
                     f['data'] = out_images[t, beg:beg+batch_size, :3, :, :]
                     f['label'] = out_normals[beg:beg+batch_size]
                     f['clusters'] = clusters_with0
                beg += batch_size
                f_test.write(test_filename+str(i)+'.h5' + '\n')

            f_test.close()


