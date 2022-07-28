from gradlib.scalar import Scalar
from gradlib.viz import draw

x = Scalar(12)
y = Scalar(5)
z = ((x * 2)**2) + (y/5)
z.backward()

dot = draw(z, show_grad=True)
dot.render('docs/graph')