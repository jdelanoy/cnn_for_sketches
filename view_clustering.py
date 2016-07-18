from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#path='./NYU_dataset/';
#clusters = np.load(path+'clusters.npy');
xc, yc, zc = (clusters_normal*2-1).T
xc2, yc2, zc2 = normal.T
#xc, yc, zc = kmeans.cluster_centers_.T

xc, yc, zc = (clusters_normal*2-1).T
#build a sphere
u_sp = np.linspace(0, 2 * np.pi, 25)
v_sp = np.linspace(0, np.pi, 20)
x_sp =  np.outer(np.cos(u_sp), np.sin(v_sp))
y_sp =  np.outer(np.sin(u_sp), np.sin(v_sp))
z_sp =  np.outer(np.ones(np.size(u_sp)), np.cos(v_sp))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xc2, yc2, zc2, color='r')
#ax.scatter(xc, yc, zc, color='b')
ax.plot_wireframe(x_sp, y_sp, z_sp, rstride=1, cstride=1, color='b', linewidth=0.1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
probas=proba_normal[:,x,y]
max_p = np.max(probas)
max_i = np.argmax(probas)
prob=np.multiply(probas,1/max_p)
ax.scatter(xc, yc, zc, c=prob, cmap='binary')
ax.plot_wireframe(x_sp, y_sp, z_sp, rstride=1, cstride=1, color='b', linewidth=0.1)

plt.plot(x_sp, y_sp, color='b', linewidth=0.1)
plt.imshow(out_normals/2+0.5)
for p in range(clusters_normal.shape[0]-1):
    plt.plot(clusters_normal[p,0]*200, clusters_normal[p,1]*200, 'o', c=str(1-prob[p]))

plt.show()

