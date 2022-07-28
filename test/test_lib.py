import unittest
from gradlib.scalar import Scalar
from micrograd.engine import Value

class TestScalar(unittest.TestCase):
  def test_gradient_calculation(self):
    x = Scalar(5)
    y = Scalar(7)
    z = x*y+x**5-y/5
    z.backward()
    a = Value(5)
    b = Value(7)
    c = a*b+a**5-b/5
    c.backward()
    self.assertEqual(z.grad, c.grad)
    self.assertEqual(x.grad, a.grad)
    self.assertEqual(y.grad, b.grad)
    
if __name__ == '__main__':
    unittest.main()
