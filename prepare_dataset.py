from dataset_tools import *
from skimage import io
from skimage import transform
from skimage import util

#principal script

nb_img = 1449;
nb_clusters = 20;
img_size = [228.0,304.0];
norm_size = [22.0,29.0];

path='./NYU_dataset/';
count = 0;

normal_path=path + 'raw_data/normals/';
depth_path=path + 'raw_data/depths/';
image_path=path + 'raw_data/images/';

#sample cluster for normals
clusters = clustering_k_means(nb_img, path + 'raw_data/normals_resize/')#random_dir(nb_clusters);
#clusters = np.load(path+'clusters.npy');
#clusters = (clusters+1)/2;
print 'Clustering done'

for i in range(nb_img):

    image_name = pad_string_with_0(i) + '.png';

    #open images
    normal_img = io.imread(normal_path + image_name );
    depth_img = io.imread(depth_path + image_name );
    image_img = io.imread(image_path + image_name );

    #build normal classification
    #TODO function for resize/crop
    normal_r = transform.rescale(normal_img, norm_size[0]/normal_img.shape[0])
    pad_h1 = round((normal_r.shape[0]-norm_size[0])/2);
    pad_h2 = (normal_r.shape[0]-norm_size[0])-pad_h1;
    pad_w1 = round((normal_r.shape[1]-norm_size[1])/2);
    pad_w2 = (normal_r.shape[1]-norm_size[1])-pad_w1;
    normal_r = util.crop(normal_r, [[pad_h1,pad_h2], [pad_w1, pad_w2], [0,0]]);
    normal_classif = cluster_normals(normal_r, clusters);

    #resize/save images
    depth_r = transform.rescale(depth_img, img_size[0]/depth_img.shape[0])
    image_r = transform.rescale(image_img, img_size[0]/image_img.shape[0])
    pad_h1 = round((image_r.shape[0]-img_size[0])/2);
    pad_h2 = (image_r.shape[0]-img_size[0])-pad_h1;
    pad_w1 = round((image_r.shape[1]-img_size[1])/2);
    pad_w2 = (image_r.shape[1]-img_size[1])-pad_w1;
    depth_r = util.crop(depth_r, [[pad_h1,pad_h2], [pad_w1, pad_w2]]);
    image_r = util.crop(image_r, [[pad_h1,pad_h2], [pad_w1, pad_w2], [0,0]]);
    io.imsave(path+'data/depth_'+image_name, depth_r);
    io.imsave(path+'data/image_'+image_name, image_r);

    #save normal classification
    io.imsave(path+'data/normal_RGB_'+image_name, normal_r);
    #np.save(path+'data/normal_'+pad_string_with_0(i), normal_classif)
    io.imsave(path+'data/normal_'+image_name, np.argmax(normal_classif,2));
    io.imsave(path+'data/normalRGB_'+image_name,
              clusters[np.argmax(normal_classif,2)]);

#save clustering
np.save(path+'clusters', clusters)

