from dataset_tools import *
from skimage import io
from skimage import transform
from skimage import util

#principal script

nb_img = 1449;
nb_clusters = 20;
img_size = [240.0, 320.0];
norm_size = [22.0,29.0];
#img_size = [228.0,304.0];
#norm_size = [22.0,29.0];

path='./NYU_dataset/';
count = 0;

normal_path=path + 'raw_data/normals/';
depth_path=path + 'raw_data/depths/';
image_path=path + 'raw_data/images/';

#sample cluster for normals
#clusters = clustering_k_means(nb_img, path + 'raw_data/normals_resize/')
clusters = np.load(path+'clusters.npy');

print 'Clustering done'

for i in range(nb_img):

    image_name = pad_string_with_0(i) + '.png';

    #open images
    normal_img = io.imread(normal_path + image_name );
    depth_img = io.imread(depth_path + image_name );
    image_img = io.imread(image_path + image_name );

    #build normal classification
    #TODO function for resize/crop
    normal_r = transform.rescale(normal_img, img_size[0]/normal_img.shape[0])
    normal_r = central_crop(normal_r, img_size);
    normal_r2 = transform.rescale(normal_img, norm_size[0]/normal_img.shape[0])
    normal_r2 = central_crop(normal_r2, norm_size);

    normal_classif = cluster_normals(normal_r2*2-1, clusters);

    #resize/save images
    depth_r = transform.rescale(depth_img, img_size[0]/depth_img.shape[0])
    image_r = transform.rescale(image_img, img_size[0]/image_img.shape[0])
    depth_r = central_crop(depth_r, img_size);
    image_r = central_crop(image_r,img_size);
    io.imsave(path+'data/depth_'+image_name, depth_r);
    io.imsave(path+'data/image_'+image_name, image_r);

    #save normal classification
    io.imsave(path+'data/normal_RGB_'+image_name, normal_r);
    #np.save(path+'data/normal_'+pad_string_with_0(i), normal_classif)
    io.imsave(path+'data/normal_'+image_name, np.argmax(normal_classif,2));
    io.imsave(path+'data/normal_classif_'+image_name,
              clusters[np.argmax(normal_classif,2)]/2.0+0.5);

#save clustering
np.save(path+'clusters', clusters)

