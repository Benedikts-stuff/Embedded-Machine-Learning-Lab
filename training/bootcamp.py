


# Task 1:
# 1.1
import torch

int_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device='cpu')
float_tensor = torch.tensor([1.0,2.0,3.0], dtype=torch.float32, device='cpu')

d2_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.int8, device='cpu')

print("Integer Tensor has type:", int_tensor.dtype, "and device:", int_tensor.device, "and shape ", int_tensor.shape)
print("Float Tensor has type:", float_tensor.dtype, "and device:", float_tensor.device, "and shape ", float_tensor.shape)
print("Float Tensor has type:", d2_tensor.dtype, "and device:", d2_tensor.device, "and shape ", d2_tensor.shape)

# 1.2
d3_tensor = torch.tensor([[[1,1], [2,2], [3,3]], [[4,4], [5,5], [6,6]], [[7,7], [8,8], [9,9]], [[11,11], [22,22], [33,33]]], dtype=torch.int8, device='cpu')
print("Float Tensor has type:", d3_tensor.dtype, "and device:", d3_tensor.device, "and shape ", d3_tensor.shape)

second_dim_elems = d3_tensor.reshape(12, 2)
print(second_dim_elems)

# format is [start:end:step], if we dont pass start its 0, if we dont pass end ist length of array
snd_elem = d3_tensor.reshape(24)[::2]
print(snd_elem)

half = d3_tensor[0:int(d3_tensor.size(0)/2):1]
print(half)
elems_half = half.reshape(12)
print(elems_half)

#3 view vs reshape

with_view = d3_tensor.view(4,6)
with_reshape = d3_tensor.reshape(4,6)
print("With view ", with_view)
print("With reshape ", with_reshape)

# Multiply and broadcasting
A = torch.tensor([[1],[2],[3]], dtype=torch.int8)
B = torch.tensor([4,5,6], dtype=torch.int8)

result = torch.multiply(A, B)
print("Multiplication result: ", result)
print("A transposed: ", A.reshape(1,3))

# permutate
tens = torch.tensor([[1,2],[2,3],[4,5]], dtype=torch.int8)
print("tensor befor permutation: ", tens)
permuted_tensor = torch.permute(tens, (1,0))
print("tensor after permutation: ", permuted_tensor)

#-----------------------------------------------------------------------------------------------------
#------------------------------------------------ Task 2 ---------------------------------------------
#-----------------------------------------------------------------------------------------------------

# 2.1
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)
z = (x**2 + 3*y)**3    # shape (3,)

# grad_outputs muss shape von z haben
#z.backward(torch.ones_like(z))
#print(x.grad)   # shape (3,)  -> elementweise 6*x*(x^2+3y)^2
#print(y.grad)   # shape (3,)  -> elementweise 9*(x^2+3y)^2

grads = torch.autograd.grad(z, (x, y), grad_outputs=torch.ones_like(z))
print(grads[0], grads[1])

# 2.2
from torch import nn
W = nn.Linear(3,2)
b = torch.ones(2, requires_grad=True)
x = torch.tensor([4.,5.,6.], requires_grad=True)

ytt = W(x) + b
ytt.backward(torch.ones_like(ytt), retain_graph=True)
ytt.backward(torch.ones_like(ytt), retain_graph=True)
print(x.grad)

# 2.3
# torch verwirft den graph
try:
    ytt.backward(torch.ones_like(ytt))
    print(x.grad)
except RuntimeError as e:
    print("Graph wurde verworfen \n:", e)