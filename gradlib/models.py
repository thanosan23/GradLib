from gradlib.scalar import Scalar
import numpy as np

class Regressor():
  def __init__(self):
    self.w = Scalar(0)
    self.b = Scalar(0)
    self.parameters = [self.w, self.b]
  def forward(self, x):
    if isinstance(x, (np.ndarray, list)):
      return np.array([(i*self.w)+self.b for i in x])
    else:
      return (x*self.w) + self.b
