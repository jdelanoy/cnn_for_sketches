###Caffe network
predict center pixel normal
`solver_center.prototxt`

`train_coarse_center.prototxt`

`deploy_coarse_center.prototxt`

predict full normal map
`solver.prototxt`

`train_coarse_full.prototxt`

`deploy_coarse.prototxt`

###Creating the database
`dataset_tools.py` contains function helping the preparation of the dataset (clustering of normals principally)

`prepare_dataset.py` : load images and prepare them for the network (resize to the right dimensions, cluster normals). Save images in path/data: depth_XXXX.png (depth map), image_XXXX.png (RGB picture), normal_RGB_XXX.png (resized normal map), normal_XXXX.png (clustered normal map, 1 channel containing index of cluster), normalRGB_XXXX.png (clustered normals written in RGB : each normal represented by its center of cluster) and save the clustering in path/clusters.npy

`prepare_dataset2.py` : load images and save them in hdf5 files (train_dataset.h5 and test_dataset.h5) , separating training and testing data

`augment_data.py` : load train_dataset.h5 and test_dataset.h5 and prepare datests for training: augment training data (crop, scaling, color change) and do clustering. Save databases train_normal_centerX.h5, train_normal_fullX.h5 (2000images per dataset), test_normal_center.h5, test_normal_full.h5,

`build_dataset.py` : load images created by `prepare_dataset.py` and create hdf5 databases : train_database.h5 containing the train set (images+clustered normals), test_database containing the test set (images+clustered normals+ground truth normals)

`view_clustering.py`: visualize the clusters on a sphere

###Testing the network
`vis_square.py` : function copied from caffe tutorials, help to visualize the parameters of conv layers

`test_network.py` : load the non trained network and train it while visualizing output or parameters of the layers.

`visualize_results.py` : load the trained caffe network and test it with the test set. For each image is shown: the input picture, the ground truth normal map (no clustering), the ground truth clustered normals and the output of the network.
