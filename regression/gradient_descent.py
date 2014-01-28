import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import numpy as np

def gradient_descent(X, y, theta, cost_function, alpha, iters):

  print("Running gradient descent with alpha =", alpha, "for", iters, "iterations.")

  m = X.shape[0]
  cost_history = []

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

  else:
    #create plot
    fig, axes = plt.subplots()
    x = np.arange(0, len(cost_history))
    y = cost_history
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Cost")
    axes.set_title("Iterations vs Cost Function with alpha=" + str(alpha))
    plt.plot(x, y)
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
          "cost_history": cost_history}

