

# a = torch.tensor([1.0])
# a.requires_grad = True  # 或者 a.requires_grad_()
# print(a)
# print(a.data)
# print(a.type())  # a的类型是tensor
# print(a.data.type())  # a.data的类型是tensor
# print(a.grad)
# print(type(a.grad))


import torch
from sympy import print_glsl

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor  乘法被重载了，相当于 tensor * tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True 计算之后计算图就回释放
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor.所以要取data，这样不会生成计算图。此时是一个纯数值的修改

        w.grad.data.zero_()  # after update, remember set the grad to zero  权重梯度数据清零，不清零的话，导数还会在。w的数据保存，但是梯度清零

    print('progress:', epoch, l.item())  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）

print("predict (after training)",4,forward(4).item())


