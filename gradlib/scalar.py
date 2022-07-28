class Scalar:
  def __init__(self, value, children=(), op=None):
    self.value = value
    self.grad = 0
    self._backward = lambda : None # local gradient * upstream gradient
    self.children = set(children)
    self.op = op
    
  def __neg__(self):
    return self * -1
    
  def backward(self):
    toposort = []
    vis = set()
    def dfs(v):
      if v not in vis:
        vis.add(v)
        if type(v) != int:          
          for child in v.children:
            dfs(child)
          toposort.append(v)
    dfs(self)   
    self.grad = 1
    toposort.reverse()
    for v in toposort:
      v._backward()
  
  def __repr__(self):
    return f"<Scalar value={self.value} grad={self.grad}>"
