class SGD:
  def __init__(self, params, lr=1e-5):
    self.lr = lr
    self.params = params
  def zero_grad(self):
    for param in self.params:
      param.grad = 0
  def step(self):
    for param in self.params:
      param.value -= self.lr * param.grad
    