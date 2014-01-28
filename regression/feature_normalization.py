import numpy as np

'''
    fn(X, y) where X is a numpy matrices.
    X is the matrix of inputs.
    Each feature (or column) shall be normalized in X.
    Normalized X is returned as a numpy matrix: X_norm.
    X's mean and std are returned as numpy matrices: X_mean, X_std.
'''
def feature_normalization(X):
  m = X.shape[1]
  print("Scaling", m, "features:")
  X_mean = np.mean(X, axis=0) 
  X_std = np.std(X, axis=0) 

  X_norm = (X - X_mean)/X_std

  print("Means are:")
  print(X_mean)
  print("Standard deviations are:")
  print(X_std)
  print()

  return {"X": X_norm, 
          "X_mean": X_mean,
          "X_std": X_std,
         }
