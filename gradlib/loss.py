import numpy as np

def MSE(y_train, y_pred):
  diff = y_train - y_pred
  return np.sum(np.square(diff)) * (1.0/len(diff))
