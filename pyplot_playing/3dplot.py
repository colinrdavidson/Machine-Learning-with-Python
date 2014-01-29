import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from compute_cost import compute_cost
from read_csv import read_csv 

file = "ex1data1.txt"
data = read_csv(file)
X_dat = data['X']
y_dat = data['y']
ones = np.matrix(np.ones(X_dat.shape[0])).T

X_dat = np.hstack([ones, X_dat])

X = np.linspace(-5, 1, 30)
Y = np.linspace(-1, 2, 30)

X, Y = np.meshgrid(X,Y)

Z = []
Z_flat = []

for i in range(0, len(X)):
  Z.append([])
  Z_flat.append([])
  for j in range(0, len(Y)):
    Z[i].append(compute_cost(X_dat, y_dat, np.matrix( [X[i][j], Y[i][j]] ).T ))
    Z_flat[i].append(4.48339)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')

ax.plot_wireframe(X, Y, Z)
ax.plot_surface(X, Y, Z_flat)

ax.scatter(0, 0, compute_cost(X_dat, y_dat, np.matrix( [0, 0] ).T), c='g')
ax.scatter(-3.89530051, 1.19298539, 4.48339, c='r')

ax.set_xlabel("Theta_0")
ax.set_ylabel("Theta_1")
ax.set_zlabel("Cost")


plt.show()
