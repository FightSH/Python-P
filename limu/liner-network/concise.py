import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

#
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))

#
net = nn.Sequential(nn.Linear(2, 1))

# 正如我们在构造nn.Linear时指定输入和输出尺寸一样， 现在我们能直接访问参数以设定它们的初始值。
# 我们通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。
# 我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 损失函数 默认情况下，它返回所有样本损失的平均值。
loss = nn.MSELoss()

# 优化算法 小批量随机梯度下降算法是一种优化神经网络的标准工具
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)