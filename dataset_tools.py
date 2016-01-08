import numpy as np
import random
from skimage import io
from math import *
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans


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


#def random_crop(image, size):
#decal = [image.shape[0]-size[0],image.shape[1]-size[1]];
    #tirer decalage aleatoire
    


#def random_scaling(image):

#def random_color(image):

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


