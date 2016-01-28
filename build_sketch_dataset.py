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

files = glob.glob(path_r+'raw_data/*/depth/chairs_large_*.png')

norm_size = [22.0, 29.0];
input_size=[228.0,304.0];

nimg = len(files);
print nimg
nb_clusters = 20;

normals_resize = np.ndarray((nimg, norm_size[0], norm_size[1], 4), np.float32);
depth_resize = np.ndarray((nimg, norm_size[0], norm_size[1]), np.float32);

out_normals = np.ndarray((nimg, norm_size[0], norm_size[1]), np.uint8);
out_depth = np.ndarray((nimg, norm_size[0], norm_size[1]), np.uint8);
out_images = np.ndarray((nb_trial, nimg, 4, input_size[0], input_size[1]), np.uint8);

for i in range(nimg):
    print i
    dir_name, img_name = path.split(files[i])
    dir_name_raw = dir_name[:-5];
    dir_name_res = dir_name[:13]+'data_resized/'+dir_name[22:-5]
    
    if not path.exists(dir_name_res+'normal/'+img_name) \
       or not path.exists(dir_name_res+'depth/'+img_name) :
        print '-----------TODO'
        #read and resize line drawings
        for t in range(nb_trial):
            img = io.imread(dir_name_raw+'T'+str(t+1)+'/'+img_name)[75:-75,208:-208]
            img = transform.rescale(img, max(input_size[0]/img.shape[0], input_size[1]/img.shape[1]));
            img = central_crop(img, input_size);
            out_images[t,i] = img.transpose(2,0,1)*255.0;
            io.imsave(dir_name_res+'T'+str(t+1)+'/'+img_name, img);
            
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
            out_images[t,i] = io.imread(dir_name_res+'T'+str(t+1)+'/'+img_name).transpose(2,0,1)*255.0;
        normals_resize[i] = io.imread(dir_name_res+'normal/'+img_name)/255.0;
        depth_resize[i] = io.imread(dir_name_res+'depth/'+img_name)/255.0;
            

#cluster on normals_resize
clusters_with0 =np.ndarray((nb_clusters+1, 3), np.float32);
clusters_with0[:nb_clusters] = clustering_k_means_array(normals_resize[:,:,:,:3])
clusters_with0[nb_clusters] = [0,0,0]
for i in range(nimg):
    label = cluster_normals(normals_resize[i,:,:,:3]*2-1, clusters_with0);
    out_normals[i] = np.argmax(label,2)

#separe train/test dataset
#600 first is  test set
for t in range(nb_trial):
    train_filename = path_r +'T'+str(t+1)+'_'+'train_dataset'

    f_train = open(train_filename+'.txt', 'w')
    batch_size=1050
    nb_batch = ceil(float(nimg)/batch_size)
    for i in range(int(nb_batch)):
        with h5py.File(train_filename+str(i)+'.h5', 'w') as f:
             f['data'] = out_images[t, i*batch_size:(i+1)*batch_size, :3, :, :]
             f['label'] = out_normals[i*batch_size:(i+1)*batch_size]
             f['clusters'] = clusters_with0
        f_train.write(train_filename+str(i)+'.h5' + '\n')

    f_train.close()


    test_filename = path_r +'T'+str(t+1)+'_'+'test_dataset.h5'

    with h5py.File(test_filename, 'w') as f:
        f['data'] = out_images[t, :, :3, :, :]
        f['label'] = out_normals
        f['clusters'] = clusters_with0

    with open(path_r+'test_dataset.txt', 'w') as f:
        f.write(test_filename + '\n')


