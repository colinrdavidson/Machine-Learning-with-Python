import numpy as np

'''
    cost(X, y, theta) where X, y, theta are numpy matrices.
    X is the matrix of input variables.
    y is the vector or output variables.
    theta is the vector or parameters.
    The cost is returned as a float.
'''

def compute_cost(X, y, theta):
  #cost(X, y, theta) = (1/2m) * sum_over_examples((h_theta - y)^2)
  m = X.shape[0]
  cost = ( (X*theta - y).getT() * (X*theta - y) ) / (2.0 * m)

  return float(cost)
