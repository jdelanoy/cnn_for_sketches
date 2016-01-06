from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

path='./NYU_dataset/';
clusters = np.load(path+'clusters.npy');
xc, yc, zc = clusters.T
#xc, yc, zc = all_normals.T
#xc, yc, zc = kmeans.cluster_centers_.T

#build a sphere
u = np.linspace(0, 2 * np.pi, 25)
v = np.linspace(0, np.pi, 20)
x =  np.outer(np.cos(u), np.sin(v))
y =  np.outer(np.sin(u), np.sin(v))
z =  np.outer(np.ones(np.size(u)), np.cos(v))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xc, yc, zc, color='r')
ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color='b', linewidth=0.1)

plt.show()
