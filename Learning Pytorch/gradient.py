import torch

# Let's see how to compute gradients in pytorch

x = torch.randn(3, requires_grad=True)
print(x)

# Everytime we do an operation on x, torch computes a computational graph
# The output y has a grad_fn attribute, that helps back propagate to get the gradients
y = x + 2
print(y)

z = y*y*2
z = z.mean()
print(z)

# Calculate the gradient dz/dx 
z.backward()
print(x.grad)