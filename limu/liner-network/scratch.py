# %matplotlib inline
import random
import torch
from d2l import torch as d2l

#
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    # 生成一个形状为 (num_examples, len(w)) 的张量 X，其中每个元素都从均值为0，标准差为1的正态分布中随机抽取。
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算输入数据矩阵 X 和权重向量 w 的矩阵乘法
    y = torch.matmul(X, w) + b
    # 加上偏置误差
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# 一维张量，长度为2
true_w = torch.tensor([2, -3.4])

true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print('features:', features[0], '\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

# 定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。 每个小批量包含一组特征和标签。
def data_iter(batch_size, features, labels):
    # 计算样本数据集中样本数量，并生成样本索引
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # yield 关键字在 Python 中用于定义生成器（generator）函数。生成器是一种特殊的迭代器，可以在函数内部产生一系列的值，而不是一次性返回所有结果。
        # 使用 yield 关键字可以让函数在每次调用时生成一个值，并在生成值后暂停函数的执行，保留当前的状态，等待下一次调用时从上次暂停的地方继续执行。
        yield features[batch_indices], labels[batch_indices]

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#  模型定义
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 损失函数
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 优化算法
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
