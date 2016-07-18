import numpy as np
import random,sys
from math import *
from scipy import spatial
from sklearn.cluster import MiniBatchKMeans
from numpy.random import random_integers
from skimage import io
from skimage import util
from skimage import transform
import matplotlib.pyplot as plt

#cluster the depths of the nb_img in path normal_path in 20 clusters
def clustering_k_means_depth(depths):
   #compute numner of pixels for each image
    nb_img = depths.shape[0]
    depth_size = depths.shape[1:];
    nb_pix  = depth_size[0]*depth_size[1];
    all_depths=np.ndarray((nb_pix*nb_img,1));
    count=0
    for i in range(nb_img):
        #read and correct image
        depth_img = depths[i].reshape((nb_pix,1));
        #depth_img = depth_img*2-1;
        #depthize depths
        for j in range(nb_pix):
            if depth_img[j] < 0.99:
                all_depths[count]= log(depth_img[j])
                count += 1;
    print nb_pix*nb_img
    print count
    #clustering
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=40);
    kmeans.fit(all_depths[:count])
    return kmeans.cluster_centers_


#cluster the depths of the nb_img in path normal_path in 20 clusters
def clustering_log_depth(depths):
    logs_r=np.linspace(0,0.9,50)
    return logs_r#np.exp(logs_r)

#cluster the normals of the nb_img in path normal_path in 20 clusters
def clustering_k_means_array(normals):
   #compute numner of pixels for each image
    nb_img = normals.shape[0]
    normal_size = normals.shape[1:];
    nb_pix  = normal_size[0]*normal_size[1];
    all_normals=np.ndarray((nb_pix*nb_img,3));
    count=0
    for i in range(nb_img):
        #read and correct image
        normal_img = normals[i].reshape((nb_pix, 3));
        norm = np.linalg.norm(normal_img,axis=1);
        for j in range(nb_pix):
            if norm[j] > 0.1:
                all_normals[count]= normal_img[j]/norm[j]
                count += 1;
    print nb_pix*nb_img
    print count
    #clustering
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=50);
    kmeans.fit(all_normals[:count])
    return kmeans.cluster_centers_


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
            #min3=[0,0,0]
            #find the min
            classif_normals[l,c,np.argmin(dist)] = 1;

    return classif_normals



def compute_proba_dist_depth(depth, clusters, sigma):
    probas = np.zeros(clusters.shape[0]);
    dist = np.zeros(clusters.shape[0]);
    norm_coeff = 1/(sigma*sqrt(2*pi))
    sum_p = 0
    for c in range(clusters.shape[0]):
        #compute geodesic distance
        dist[c]=abs(clusters[c]-depth)
        probas[c] = norm_coeff*exp(-dist[c]*dist[c]/(2*sigma*sigma))
        sum_p = sum_p + (probas[c])
    if sum_p == 0:
        sys.stderr.write('=============ERROR IN PROBAS')
    probas = np.divide(probas, sum_p)
    return probas

#find the cluster for each depth
def cluster_depths_gaussian(depths, clusters):
    height=depths.shape[0];
    width=depths.shape[1];
    nb_clusters = clusters.shape[0];
    classif_depths = np.zeros((nb_clusters, height, width));
    for l in range(height):
        for c in range(width):
            if depths[l,c] > 0.99:
                classif_depths[nb_clusters-1,l,c]=1
            elif depths[l,c] < 0.015:
                classif_depths[0,l,c]=1
            else:
                classif_depths[:,l,c] = compute_proba_dist_depth(depths[l,c], clusters, 0.05)
    return classif_depths



def compute_proba_dist(normal, clusters, sigma):
    probas = np.zeros(clusters.shape[0]);
    dist = np.zeros(clusters.shape[0]);
    norm_coeff = 1/(sigma*sqrt(2*pi))
    norm_coeff = 1/(sigma*sqrt(2*pi))
    sum_p = 0
    for c in range(clusters.shape[0]):
        #compute geodesic distance
        dot = np.dot(clusters[c], normal)
        dot = max(min(dot,1.0),-1.0)
        dist[c]=acos(dot)
        probas[c] = norm_coeff*exp(-dist[c]*dist[c]/(2*sigma*sigma))
        sum_p = sum_p + (probas[c])
    probas = np.divide(probas, sum_p)
    return probas

#find the cluster for each normal
def cluster_normals_gaussian(normals, clusters):
    height=normals.shape[0];
    width=normals.shape[1];
    nb_clusters = clusters.shape[0];

    classif_normals = np.zeros((nb_clusters, height, width));

    for l in range(height):
        for c in range(width):
            if np.linalg.norm(normals[l,c]) < 0.1:
                classif_normals[nb_clusters-1,l,c]=1
            else:
                classif_normals[:,l,c] = compute_proba_dist(normals[l,c], clusters, pi/15)
    return classif_normals

def random_crop(image, normal, size):
    decal = [image.shape[0]-size[0],image.shape[1]-size[1]];
    #tirer decalage aleatoire
    decal_alea = [random_integers(0,decal[0]),random_integers(0,decal[1])]
    crop_img = image[decal_alea[0]:decal_alea[0]+size[0], decal_alea[1]:decal_alea[1]+size[1]]
    crop_normal = normal[decal_alea[0]:decal_alea[0]+size[0], decal_alea[1]:decal_alea[1]+size[1]]
    return crop_img,crop_normal

    
def central_crop(image, size):
    pad_0 = round((image.shape[0]-size[0])/2);
    pad_1 = round((image.shape[1]-size[1])/2);
    image_c = image[pad_0:pad_0+size[0], pad_1:pad_1+size[1]]
    return image_c;

def random_scaling(image, normal, size, a, b):
    scale = random.uniform(a, b);
    #print scale
    #resize images
    img_r = transform.rescale(image, scale);
    norm_r = transform.rescale(normal, scale);
    img_r, norm_r = random_crop(img_r, norm_r, size);
    #TODO modify depth : divide by scale
    #modify normals
    #for line in range(norm_r.shape[0]):
    #    for col in range(norm_r.shape[1]):
    #        norm_r[line,col,2] = norm_r[line,col,2] * scale;
    #        norm = np.linalg.norm(norm_r[line,col]);
    #        norm_r[line,col] = norm_r[line,col]/norm;
    return img_r, norm_r;
            
    
def random_color(image):
    scale = random.uniform(0.8, 1.2);
    #print scale
    image = image * scale;
    for line in range(image.shape[0]):
        for col in range(image.shape[1]):
            for c in range(image.shape[2]):
                if image[line,col,c] > 1:
                    image[line,col,c] = 1
    return image;

def trim_values(image, pad=20):
    im_size = image.shape
    height = image.shape[0]
    width = image.shape[1]

    BBOX = compute_BBOX(image)
    BBOX[0][0] -= pad
    BBOX[0][1] -= pad
    BBOX[1][0] += pad
    BBOX[1][1] += pad

    ratios = [(BBOX[1][0]-BBOX[0][0])*1.0/im_size[0], (BBOX[1][1]-BBOX[0][1])*1.0/im_size[1]]
    #find ratio
    ratio_ok = np.argmax(ratios);
    ratio_tomove = np.argmin(ratios);

    final_size = image.shape[ratio_tomove]*ratios[ratio_ok]
    to_add = int((final_size - (BBOX[1][ratio_tomove]-BBOX[0][ratio_tomove]))/2)

    BBOX[1][ratio_tomove]+=to_add
    BBOX[0][ratio_tomove]-=to_add

    return BBOX


def compute_BBOX(image):
    BBOX =[[image.shape[0], image.shape[1]], [0, 0]]
    first = False;
    last = False;
    for line in range(image.shape[0]):
        last = False;
        for col in range(image.shape[1]):
            if (image[line,col] != [255,255,255]).any():
                first = True;
                last = True;
                if line < BBOX[0][0]:
                    BBOX[0][0]= line
                if line > BBOX[1][0]:
                    BBOX[1][0] = line
                if col < BBOX[0][1]:
                    BBOX[0][1]= col
                if col > BBOX[1][1]:
                    BBOX[1][1] = col
        if first == True and last == False:
            break;
    return BBOX;

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

