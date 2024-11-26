import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None

#  2*(0*0+1*1+2*2+3*3) = 28
y = 2 * torch.dot(x, x)
print(y)

y.backward()
x.grad



