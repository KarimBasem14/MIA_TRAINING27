import numpy as np

# f = w * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

def forward(x):
    return w * x

def loss(y, y_hat):
    return ((y_hat - y)**2).mean()

# dL/dw = 1/N 2*x (y_hat - y)
def gradient(x, y, y_hat):
    return np.dot(2*x, y_hat - y).mean()


learning_rate = 0.01
n_iters = 100


for epoch in range(n_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    grad = gradient(X, Y, y_pred)

    w -= learning_rate * grad

    if epoch % 9 == 0:
        print(f"Current weight = {w}, current loss = {l}")

