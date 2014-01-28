import matplotlib
matplotlib.use('ps')
import matplotlib.pyplot as plt
import numpy as np
from read_csv import read_csv

def test():
  file = "ex1data2.txt"
  data = read_csv(file)
  X = data['X']
  y = data['y']

  fig, axes = plt.subplots(1, 2)

  axes[0].set_xlabel('Square Footage')
  axes[0].set_ylabel('Price')
  axes[0].set_title('Price by Size')
  axes[0].plot(X[:,0], y, 'bo')

  axes[1].set_xlabel('Number of Bedrooms')
  axes[1].set_ylabel('Price')
  axes[1].set_title('Price by Bedrooms')
  axes[1].plot(X[:,1], y, 'ro')

  fig.tight_layout()

  plt.savefig("fig1.ps")

test()
