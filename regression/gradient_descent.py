import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def gradient_descent(X, y, theta, cost_function, alpha, iters):

  print("Running gradient descent with alpha =", alpha, "for", iters, "iterations.")

  m = X.shape[0]
  cost_history = []
  theta_history = []

  for i in range(0, iters):
    #update theta
    theta = theta - ( (alpha/m) * (X.T * ( (X * theta) - y) ) )
    cost = cost_function(X, y, theta)

    #check for a janky cost
    if cost_history != [] and cost_history[-1] < cost:
      print("WARNING: Cost just increased, aborting gradient descent. Take a look at your learning rate, it may be too high.")
      print("ABORTING!")
      break
    
    '''
    #Currently not using this, but maybe in the future
    #Check for early convergence
    if cost_history != [] and cost_history[-1] - cost < 0.00001:
      print("Change in cost is < 1e-5, declaring convergence.")
      print("Thets is:")
      print(theta)
      break
    '''

    cost_history.append(cost)
    theta_history.append(theta)

  else:
    #create plot of cost
    fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1)
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(0, len(cost_history))
    y = cost_history
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title("Iterations vs Cost Function with alpha=" + str(alpha))
    plt.plot(x, y)

    ''' 
    #3d plot of cost function, only works with 1 feature.
    ax = fig.add_subplot(1, 2, 2, projection = '3d')
    theta_graph = [ [i[0,0], i[1,0]] for i in theta_history]
    theta_graph = np.matrix(theta_graph)
    X = theta_graph[:,0];
    Y = theta_graph[:,1];
    Z = cost_history

    ax.plot(X, Y, Z)
    '''

    plt.show()

    #Output some info on gradient descent
    print("Gradient descent completed all iterations.")
    print("Last 2 costs were:")
    print(cost_history[-2])
    print(cost_history[-1])
    print("Theta is:")
    print(theta)

  print()
  return {"theta": theta,
          "cost_history": cost_history,
          "theta_history": theta_history}
