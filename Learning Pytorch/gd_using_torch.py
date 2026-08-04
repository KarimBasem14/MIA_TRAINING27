# Pipelines steps:
# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#   - Forward pass: Compute prediction
#   - Backward pass: Gradients
#   - Update weights

# import numpy as np
import torch
import torch.nn as nn

X_test = torch.tensor([5], dtype = torch.float32)

# f = w * x
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# def forward(x):
#     return w * x

learning_rate = 0.1
n_iters = 200

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# dL/dw = 1/N 2*x (y_hat - y)
# def gradient(x, y, y_hat):
#     return np.dot(2*x, y_hat - y).mean()





for epoch in range(n_iters):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward() # Calculates dl/dw

    optimizer.step()
    # with torch.no_grad():
    #     w -= learning_rate * w.grad

    optimizer.zero_grad()

    # w.grad.zero_()

    if epoch % 9 == 0:
        [w, b] = model.parameters()
        print(f"Current weight = {w[0][0].item()}, current loss = {l}")
for x in range(6):
    print(f"Prediction after training for value {x} is {model(torch.tensor([x], dtype = torch.float32)).item()}")