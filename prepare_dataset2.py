from dataset_tools import *
from skimage import io
import h5py
import sklearn.cross_validation


nb_img = 1449;
nb_clusters = 20;
img_size = [480.0, 640.0];

path='/home/NYU_dataset/';

normal_path=path + 'raw_data/normals/';
depth_path=path + 'raw_data/depths/';
image_path=path + 'raw_data/images/';

#sample cluster for normals
#clusters = clustering_k_means(nb_img, path + 'raw_data/normals_resize/')
clusters = np.load(path+'clusters.npy');

normals = np.ndarray((nb_img, img_size[0], img_size[1], 3), np.uint8);
images = np.ndarray((nb_img, img_size[0], img_size[1], 3), np.uint8);
depths = np.ndarray((nb_img, img_size[0], img_size[1]), np.int32);


print 'Clustering done'

for i in range(nb_img):

    image_name = pad_string_with_0(i) + '.png';

    #open images
    normals[i] = io.imread(normal_path + image_name );
    depths[i] = io.imread(depth_path + image_name );
    images[i] = io.imread(image_path + image_name );



#separe train/test dataset
im, imT, norm, normT, depth, depthT= sklearn.cross_validation.train_test_split(images, normals, depths)

train_filename = path +'train_dataset.h5'
test_filename = path +'test_dataset.h5'

#write datasets
with h5py.File(train_filename, 'w') as f:
    f['images'] = im
    f['normals'] = norm
    f['depths'] = depth
    f['clusters'] = clusters
with h5py.File(test_filename, 'w') as f:
    f['images'] = imT
    f['normals'] = normT
    f['depths'] = depthT
    f['clusters'] = clusters
