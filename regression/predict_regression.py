import numpy as np

'''
    predict_regression(x, theta, mean, std) where x, mean, std are numpy matrices.
    x is a row to predict without the intercept term.
    mean and std are rows with the same dimension as x.
    The prediction based on theta is returned as a numpy matrix.
'''

def predict_regression(x, theta, mean=0, std=1):
  one = np.matrix([1])
  x_norm = (x - mean) / std
  x_norm = np.hstack([one, x_norm])

  predict = x_norm*theta

  return predict
