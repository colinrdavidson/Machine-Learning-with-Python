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

X = np.linspace(-5, -3, 100)
Y = np.linspace(0, 2, 100)

X, Y = np.meshgrid(X,Y)

Z = []

for i in range(0, len(X)):
  Z.append([])
  for j in range(0, len(Y)):
    Z[i].append(compute_cost(X_dat, y_dat, np.matrix( [X[i][j], Y[i][j]] ).T ))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter([-3.89530051], [1.19298539], 0, c='r')
ax.set_xlabel("Theta_0")
ax.set_ylabel("Theta_1")
ax.set_zlabel("Cost")

ax.plot_surface(X, Y, Z)

plt.show()






