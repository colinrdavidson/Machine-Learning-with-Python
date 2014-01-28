import csv
import numpy as np

''' 
    read_csv(file) where file is a string.
    file is the path to the csv file.
    It is assumed that there is a single y vector(for now).
    X and y are returned as numpy matrices with float entries.
'''

def read_csv(file):
  print("Reading in data from", file, ".")
  raw_data = []

  with open(file, "r") as raw_file:
    reader = csv.reader(raw_file)
    for row in reader:
      raw_data.append([float(i) for i in row])

  assert raw_file.closed == True

  raw_data = np.matrix(raw_data)

  m = raw_data.shape[0]
  n = raw_data.shape[1] - 1
  print ("This data has", n, "feature(s) in X and contains", m, "training examples.")

  X = raw_data[:,0:n]
  y = raw_data[:,n]

  print()

  return {"X": X, "y": y}
