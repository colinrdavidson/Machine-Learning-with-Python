import numpy as np
from feature_normalization import feature_normalization
from read_csv import read_csv
from compute_cost import compute_cost
from gradient_descent import gradient_descent
from predict_regression import predict_regression

#data = read_csv("ex1data2.txt")
data = read_csv("ex1data1.txt")

X = data["X"]
y = data["y"]

#normalization flag
normalize = False 

if normalize:
  #Normalize features and keep old X around for kicks
  X_old = X 
  data_norm = feature_normalization(X)
  #Get our normalized features and stats
  X = data_norm["X"]
  X_mean = data_norm["X_mean"]
  X_std = data_norm["X_std"]

m = X.shape[0]

#Pad the data with 1's for the intercept term
pad = np.matrix(np.ones(m)).T
X = np.hstack([pad, X])

n = X.shape[1] #this needs to be assigned AFTER the pad step

#In initial theta to feed into gradient descent
theta_init = np.matrix(np.zeros(n)).T

#Gradient descent params
alpha = 0.01
iters = 5000

#Run gradient descent
run_descent = gradient_descent(X=X,
                               y=y,
                               theta=theta_init,
                               cost_function = compute_cost,
                               alpha=alpha,
                               iters=iters)
theta = run_descent["theta"]
hist = run_descent["cost_history"]

#predict for ex2data2
'''
x = np.matrix([1800, 4])
predict = predict_regression(x=x,
                             theta=theta,
                             mean=X_mean,
                             std=X_std)
'''

