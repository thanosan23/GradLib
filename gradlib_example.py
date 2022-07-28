# Linear regression using custom built autograd library
import numpy as np
import matplotlib.pyplot as plt
from gradlib.models import Regressor
from gradlib.loss import MSE
from gradlib.optim import SGD

X_train = np.arange(0, 100, 1)
y_train = np.arange(0, 100, 1) * 2

sample = np.random.permutation(y_train.shape[0])
X_train = X_train[sample]
y_train = y_train[sample]

lr = 3e-4
epochs = 5000

model = Regressor()
optimizer = SGD(model.parameters, lr=lr)

losses = []

for t in range(epochs):
  # forward pass
  y_pred = model.forward(X_train)
  loss = MSE(y_train, y_pred)

  # backward pass
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  losses.append(loss.value)
  print(t+1, loss.value)

X_test = np.array([1234, 2048, 4092])
y_test = X_test * 2
y_pred = model.forward(X_test)
print(f"Expected: {y_test}")
print(f"Predicted: {[yi.value for yi in y_pred]}")

print(f"Weight: {model.w.value}")
print(f"Bias: {model.b.value}") 

plt.title("Loss over time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(losses)
plt.savefig("docs/loss.svg")