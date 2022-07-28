from gradlib.scalar import Scalar
from gradlib.function import Function

def add_operation(name, fn):
  assert isinstance(fn, Function), "2nd argument must be of type Function"
  setattr(Scalar, f"__{name}__", lambda self, oth : fn.do_op(self, oth))
  setattr(Scalar, f"__r{name}__", lambda self, oth : fn.do_op(oth, self))

# ******* ADDITION *********

add = Function('+')

def add_forward(cur, oth):
  return cur.value + oth.value
  
def add_backward(cur, oth, out):
  cur.grad += 1 * out.grad
  oth.grad += 1 * out.grad
  return cur.grad, oth.grad, out.grad

add.set_fns(add_forward, add_backward)

# ******* MULTIPLICATION *********

mul = Function('*')

def mul_forward(cur, oth):
  return cur.value * oth.value

def mul_backward(cur, oth, out):
  cur.grad += oth.value * out.grad
  oth.grad += cur.value * out.grad
  return cur.grad, oth.grad, out.grad

mul.set_fns(mul_forward, mul_backward)

# ******* EXPONENTIATION *********

pow = Function('^', oth_is_scalar=False)

def pow_forward(cur, oth):
  assert isinstance(oth, (int, float)), "Exponentiation only supports integer and float powers"
  return cur.value ** oth 
  
def pow_backward(cur, oth, out):
  cur.grad += (oth * cur.value**(oth-1)) * out.grad
  return cur.grad, out.grad

pow.set_fns(pow_forward, pow_backward)

# ******* SUBTRACTION *********

sub = Function('-')

def sub_forward(cur, oth):
  return cur.value + (-oth.value)

def sub_backward(cur, oth, out):
  cur.grad += 1 * out.grad
  oth.grad += -1 * out.grad
  return cur.grad, oth.grad, out.grad

sub.set_fns(sub_forward, sub_backward)

# ******* DIVISION *********

truediv = Function('/', oth_is_scalar=False, backward_needed=False)

def truediv_forward(cur, oth):
  assert oth != 0, "Can't divide by 0!"
  return cur * oth**-1

truediv.set_fns(truediv_forward)

ops = {"add" : add, "mul" : mul, "pow" : pow, "sub" : sub, "truediv" : truediv}
for name, op in ops.items():
  add_operation(name, op)