import numpy as np
import random
from math import *
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans
from numpy.random import random_integers
from skimage import io
from skimage import util
from skimage import transform
import matplotlib.pyplot as plt

#cluster the normals of the nb_img in path normal_path in 20 clusters
def clustering_k_means(nb_img, normal_path):
    #compute numner of pixels for each image
    normal_size = io.imread(normal_path + '0000.png').shape;
    nb_pix  = normal_size[0]*normal_size[1];
    all_normals=np.ndarray((nb_pix*nb_img,3));

    for i in range(nb_img):
        #read and correct image
        image_name = pad_string_with_0(i) + '.png';
        normal_img = io.imread(normal_path + image_name )[:,:,:3].reshape((nb_pix, 3))/255.0;
        normal_img = normal_img*2-1;
        #normalize normals
        norm = np.linalg.norm(normal_img,axis=1);
        for j in range(nb_pix):
            if norm[j] > 0.1:
                normal_img[j]= normal_img[j]/norm[j]
        all_normals[nb_pix*i:nb_pix*(i+1)] = normal_img;
    #clustering
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=20);
    kmeans.fit(all_normals)
    return kmeans.cluster_centers_

#find the cluster for each normal
def cluster_normals(normals, clusters):
    height=normals.shape[0];
    width=normals.shape[1];
    nb_clusters = clusters.shape[0];
    
    classif_normals = np.zeros((height, width, nb_clusters));
    
    for l in range(height):
        for c in range(width):
            #compute all distances
            dist = spatial.distance_matrix(clusters, np.reshape(normals[l, c, :],(1,3)));
            min3=[0,0,0]
            #find the min
            classif_normals[l,c,np.argmin(dist)] = 1;

    return classif_normals


def random_crop(image, normal, size):
    decal = [image.shape[0]-size[0],image.shape[1]-size[1]];
    #tirer decalage aleatoire
    decal_alea = [random_integers(0,decal[0]),random_integers(0,decal[1])]
    crop_img = util.crop(image, [[decal_alea[0],decal[0]-decal_alea[0]], [decal_alea[1],decal[1]-decal_alea[1]], [0,0]]);
    #crop_depth = util.crop(image, [[decal_alea[0],decal[0]-decal_alea[0]], [decal_alea[1],decal[1]-decal_alea[1]]]);
    crop_normal = util.crop(normal, [[decal_alea[0],decal[0]-decal_alea[0]], [decal_alea[1],decal[1]-decal_alea[1]], [0,0]]);
    return crop_img,crop_normal

    
def central_crop(image, size):
    pad_h1 = round((image.shape[0]-size[0])/2);
    pad_h2 = (image.shape[0]-size[0])-pad_h1;
    pad_w1 = round((image.shape[1]-size[1])/2);
    pad_w2 = (image.shape[1]-size[1])-pad_w1;
    if image.ndim == 2:
        image = util.crop(image, [[pad_h1,pad_h2], [pad_w1, pad_w2]]);
    elif image.ndim == 3:
        image = util.crop(image, [[pad_h1,pad_h2], [pad_w1, pad_w2], [0,0]]);
    return image;

def random_scaling(image, normal, size):
    scale = random.uniform(1.0, 1.5);
    print scale
    #resize images
    img_r = transform.rescale(image, scale);
    img_r = central_crop(img_r, size);
    norm_r = transform.rescale(normal, scale);
    norm_r = central_crop(norm_r, size);
    #TODO modify depth : divide by scale
    #modify normals
    for line in range(norm_r.shape[0]):
        for col in range(norm_r.shape[1]):
            norm_r[line,col,2] = norm_r[line,col,2] * scale;
            norm = np.linalg.norm(norm_r[line,col]);
            norm_r[line,col] = norm_r[line,col]/norm;
    return img_r, norm_r;
            
    
def random_color(image):
    scale = random.uniform(0.8, 1.2);
    print scale
    image = image * scale;
    for line in range(image.shape[0]):
        for col in range(image.shape[1]):
            for c in range(image.shape[2]):
                if image[line,col,c] > 1:
                    image[line,col,c] = 1
    return image;
    
#compute n random view directions
def random_dir(n=1):
    dirs=np.zeros((n,3));
    for i in range(n):
        u1=random.random();
        u2=random.random();
        z = 1 - 2 * u1;
        r = sqrt(max(0, 1 - z * z));
        phi = 2 * pi * u2;
        x = r * cos(phi);
        y = r * sin(phi);
        dirs[i,:] = [x,y,z];
        #dirs[i,:] = dirs[i,:] / norm(dirs[i,:]);
    return dirs;

def nb_0_string(i):
    if i<10:
        n = 3;
    elif i<100:
        n=2;
    elif i<1000:
        n=1;
    else:
        n=0;
    return n;

#return a string of size 4 padded with 0 repreenting number i
def pad_string_with_0(i):
    nb = nb_0_string(i);
    padded = str(i);
    for n in range(nb):
       padded = '0'+padded;
    return padded;


