from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import *

caffe_root = '/user/delanoy/home/caffe-future/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe



def generate_normal_circle(size):
    out_normals = np.ones((size, size,3), np.float16);
    for i in  range(size):
        for j in range(size):
            #se ramener entre -1 et 1
            x = i*2.0/float(size)-1
            y = j*2.0/float(size)-1
            if x*x+y*y<1.0:
                z = sqrt(1-x*x-y*y)
                out_normals[i,j] = [y,x,z]
    return out_normals
    
def compute_probas(net_output):
    proba = np.array(net_output, dtype=np.float32)
    for l in range(proba.shape[1]):
        for c in range(proba.shape[2]):
            proba[:,l,c] = np.exp(net_output[:,l,c])/np.sum(np.exp(net_output[:,l,c]))
    return net_output

def compute_proba_tex(dataset, net, img, save=True, show=False, save_path=''):
    image = dataset['data'][img];
    normal = dataset['label_normal'][img];
    clusters_normal = dataset['clusters_normal'][:,:]/2+0.5;
    depth = dataset['label_depth'][img];
    clusters_depth = np.exp(dataset['clusters_depth'][:]);

    #load a test image
    net.blobs['data'].data[...] = image;
    out = net.forward();
    #depth
    proba_depth = compute_probas(out['output_depth'][0])
    proba_normal = compute_probas(out['output_normal'][0])

    depth_proba=np.ndarray([550,370])
    depth_proba_gt=np.ndarray([550,370])
    normal_map= clusters_normal[np.argmax(proba_normal,0).astype(int)]
    normal_map_gt= clusters_normal[np.argmax(normal,0).astype(int)]
    for c in range(clusters_depth.shape[0]-1):
        x=c%5*74
        y=c/5*55
        depth_proba[y:y+55,x:x+74]=proba_depth[c]
        depth_proba_gt[y:y+55,x:x+74]=depth[c]

    #if show:
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ###
    plt.subplot(1,5,1)
    plt.imshow(depth_proba,cmap='gray', norm = norm)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.colorbar(fraction=0.05, shrink=0.8)
    ###
    plt.subplot(1,5,2)
    plt.imshow(depth_proba_gt,cmap='gray', norm = norm)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.colorbar(fraction=0.05, shrink=0.8)
    norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
    ###
    plt.subplot(1,5,3)
    plt.imshow(normal_map)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(1,5,4)
    plt.imshow(normal_map_gt)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(1,5,5)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.show()
    if save:
        io.imsave(save_path+'depth_'+str(img)+'.png',depth_proba)
        io.imsave(save_path+'depth_'+str(img)+'_gt.png',depth_proba_gt)
        io.imsave(save_path+'normal_'+str(img)+'.png',normal_map)
        io.imsave(save_path+'normal_'+str(img)+'_gt.png',normal_map_gt)


        
def show_prediction_sketch(dataset, net, img_name, save=False, show=True, save_path=''):
    out_normals = generate_normal_circle(200)

    clusters_normal = dataset['clusters_normal'][:,:]/2+0.5;
    clusters_depth = np.exp(dataset['clusters_depth'][:]);

    img = io.imread(img_name).transpose(2,0,1)[:3]/255.0
    net.blobs['data'].data[...] = img
    out = net.forward();

    proba_depth=compute_probas(out['output_depth'][0])
    confidence_depth = np.max(proba_depth,0)
    proba_normal=compute_probas(out['output_normal'][0])
    confidence_normal = np.max(proba_normal,0)
    
    #visualization of results
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ###
    plt.subplot(2, 3, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 3, 4)
    plt.imshow(out_normals/2+0.5)
    plt.title('normals')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 3, 2)
    plt.imshow(clusters_normal[np.argmax(proba_normal,0).astype(int)])
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 3, 3)
    plt.imshow(confidence_normal, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 3, 5)
    plt.imshow(clusters_depth[np.argmax(proba_depth,0).astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 3, 6)
    plt.imshow(confidence_depth, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()







def show_prediction_dataset(dataset, net, img, save=False, show=True, save_path=''):
    image = dataset['data'][img];
    normal = dataset['label_normal'][img];
    clusters_normal = dataset['clusters_normal'][:,:]/2+0.5;
    depth = dataset['label_depth'][img];
    clusters_depth = dataset['clusters_depth'][:];
    #load a test image
    net.blobs['data'].data[...] = image;
    out = net.forward();
    proba_depth=compute_probas(out['output_depth'][0])
    confidence_depth = np.max(proba_depth,0)
    proba_normal=compute_probas(out['output_normal'][0])
    confidence_normal = np.max(proba_normal,0)

    #visualization of results
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ###
    plt.subplot(2, 4, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 2)
    plt.imshow(clusters_normal[np.argmax(normal,0).astype(int)])
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 3)
    plt.imshow(clusters_normal[np.argmax(proba_normal,0).astype(int)])
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 4)
    plt.imshow(confidence_normal, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 5)
    plt.imshow(generate_normal_circle(200)/2+0.5)
    plt.title('normals')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 6)
    plt.imshow(clusters_depth[np.argmax(depth,0).astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 7)
    plt.imshow(clusters_depth[np.argmax(proba_depth,0).astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 8)
    plt.imshow(confidence_depth, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()



def show_prediction_pair(dataset, net, img, save=False, show=True, save_path=''):
    image = dataset['data'][img];
    normal = dataset['label_normal'][img];
    clusters_normal = dataset['clusters_normal'][:,:]/2+0.5;
    depth = dataset['label_depth'][img];
    depth_pred = dataset['depth_prediction'][img];
    clusters_depth = dataset['clusters_depth'][:];
    #load a test image
    net.blobs['data'].data[...] = image;
    out = net.forward();
    proba_depth=compute_probas(out['output_depth'][0])
    confidence_depth = np.max(proba_depth,0)
    proba_normal=compute_probas(out['output_normal'][0])
    confidence_normal = np.max(proba_normal,0)

    #visualization of results
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ###
    plt.subplot(2, 4, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 2)
    plt.imshow(clusters_normal[np.argmax(normal,0).astype(int)])
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 3)
    plt.imshow(clusters_normal[np.argmax(proba_normal,0).astype(int)])
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 4)
    plt.imshow(confidence_normal, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 5)
    plt.imshow(clusters_depth[depth_pred.astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('depth_pred')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 6)
    plt.imshow(clusters_depth[np.argmax(depth,0).astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 7)
    plt.imshow(clusters_depth[np.argmax(proba_depth,0).astype(int)], cmap='jet', norm = norm) #add [:,:,0] for depth
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 4, 8)
    plt.imshow(confidence_depth, cmap='jet', norm = norm)
    plt.title('confidence')
    frame1 = plt.gca()
    plt.colorbar(fraction=0.05, shrink=0.8)
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()




    
#show proba distrib of prediction for image img, at pixel (x,y)
def show_proba_distrib(dataset, net, img, x_pix, y_pix, save=False, show=True, save_path=''):
    out_normals = generate_normal_circle(200)
    image = dataset['data'][img];
    normal = dataset['label_normal'][img];
    clusters_normal = dataset['clusters_normal'][:,:]/2+0.5;
    depth = dataset['label_depth'][img];
    clusters_depth = np.exp(dataset['clusters_depth'][:]);

    #load a test image
    net.blobs['data'].data[...] = image;
    out = net.forward();
    proba_depth=compute_probas(out['output_depth'][0])
    confidence_depth = np.max(proba_depth,0)
    proba_normal=compute_probas(out['output_normal'][0])
    confidence_normal = np.max(proba_normal,0)

    #visualization of results
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    ###
    plt.subplot(2, 5, 1)
    plt.imshow(net.blobs['data'].data[0].transpose(1,2,0))
    plt.title('input_image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 2)
    plt.imshow(clusters_normal[np.argmax(normal,0).astype(int)])
    plt.plot(y_pix,x_pix,'sr', scalex=False, scaley=False, markeredgecolor='r', markerfacecolor="None", markeredgewidth=2);
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 3)
    plt.imshow(out_normals/2+0.5)
    probas=normal[:,x_pix,y_pix]
    prob=np.multiply(probas,1/np.max(probas))
    for p in range(clusters_normal.shape[0]-1):
        plt.plot(clusters_normal[p,0]*200, clusters_normal[p,1]*200, 'o', c=str(1-prob[p]), scalex=False, scaley=False)

    plt.title('confidence gt')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 4)
    plt.imshow(clusters_normal[np.argmax(proba_normal,0).astype(int)])
    plt.plot(y_pix,x_pix,'sr', scalex=False, scaley=False, markeredgecolor='r', markerfacecolor="None", markeredgewidth=2);
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 5)
    plt.imshow(out_normals/2+0.5)
    probas=proba_normal[:,x_pix,y_pix]
    prob=np.multiply(probas,1/np.max(probas))
    for p in range(clusters_normal.shape[0]-1):
        plt.plot(clusters_normal[p,0]*200, clusters_normal[p,1]*200, 'o', c=str(1-prob[p]), scalex=False, scaley=False)

    plt.title('confidence')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 7)
    plt.imshow(clusters_depth[np.argmax(depth,0).astype(int)], cmap='jet', norm = norm)
    plt.plot(y_pix,x_pix,'sr', scalex=False, scaley=False, markeredgecolor='r', markerfacecolor="None", markeredgewidth=2);
    plt.title('ground_truth')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    ###
    plt.subplot(2, 5, 8)
    plt.title('confidence gt')
    probas=depth[:,x_pix,y_pix]
    plt.plot(clusters_depth,probas, 'o')
    ###
    plt.subplot(2, 5, 9)
    plt.imshow(clusters_depth[np.argmax(proba_depth,0).astype(int)], cmap='jet', norm = norm)
    plt.plot(y_pix,x_pix,'sr', scalex=False, scaley=False, markeredgecolor='r', markerfacecolor="None", markeredgewidth=2);
    plt.title('network_output')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.subplot(2, 5, 10)
    plt.title('confidence')
    probas=proba_depth[:,x_pix,y_pix]
    plt.plot(clusters_depth,probas, 'o')
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
