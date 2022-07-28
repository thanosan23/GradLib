from gradlib.scalar import Scalar

class Function:
  def __init__(self, op, oth_is_scalar=True, backward_needed=True):
    self.forward = None
    self.backward = None
    self.op = op
    self.oth_is_scalar = oth_is_scalar
    self.backward_needed = backward_needed 

  def set_fns(self, forward, backward= None):
    self.forward = forward
    if backward != None:
      self.backward = backward
    
  def do_op(self, cur, oth):
    if self.oth_is_scalar:
      oth = oth if isinstance(oth, Scalar) else Scalar(oth)
      children = (cur, oth)
    else:
      children = (cur,)
      
    if self.backward_needed:
      cur = cur if isinstance(cur, Scalar) else Scalar(cur)
      out = Scalar(self.forward(cur, oth), children=children, op=(self.op if self.oth_is_scalar else self.op+str(oth)))
      def _backward():
        if self.oth_is_scalar:
          cur.grad, oth.grad, out.grad = self.backward(cur, oth, out)
        else:
          cur.grad, out.grad = self.backward(cur, oth, out)
      out._backward = lambda : _backward()
      return out
    else:
      return self.forward(cur, oth)