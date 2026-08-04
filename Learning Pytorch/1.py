import torch

# Everything in pytorch is a tensor (an object)

# Make a tensor with zeros
# x = torch.empty(1)  size/dimensions
# print(x)

# x = torch.empty(3, 2)  3 rows and 2 colms
# print(x)

# # Rand values
# x = torch.rand(3, 2)
# print(x)

# print(x.size())

# # Make a tensor from an array
# x = torch.tensor([2.5, 0.1])


# OPERATIONS

x = torch.rand(2, 2)
y = torch.rand(2,2)
print(x)
print(y)

# Element wise additions
addition = x + y
addition = torch.add(x, y)
print(addition)


# Edits y
# Every function with _ means inplace
y.add_(x)
print(y)