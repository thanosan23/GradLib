# Linear regression from scratch (without autograd)
import numpy as np

X_train = np.arange(0, 100, 1)
y_train = np.arange(0, 100, 1) * 2

sample = np.random.permutation(y_train.shape[0])
X_train = X_train[sample]
y_train = y_train[sample]

w = 0
b = 0

lr = 3e-4

mse = lambda diff : np.sum(np.square(diff)) * (1.0/len(diff))

for t in range(1000):
  y_pred = X_train * w + b
  diff = y_train - y_pred
  loss = mse(diff)

  grad_w = -2 * X_train.dot(diff).sum() * (1.0/len(X_train))
  grad_b = -2 * diff.sum() * (1.0/len(X_train))

  w -= lr * grad_w
  b -= lr * grad_b
  print(t+1, loss)

X_test = np.array([1234, 2048, 4092])
y_test = X_test * 2
y_pred = X_test * w + b
print(f"Expected: {y_test}")
print(f"Predicted: {y_pred}")

print(f"Weighs: {w}")
print(f"Bias: {b}") 
