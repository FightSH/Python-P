import torch


# 张量表示一个由数值组成的数组，这个数组可能有多个维度。
# 具有一个轴的张量对应数学上的向量（vector）；
# 具有两个轴的张量对应数学上的矩阵（matrix）；
# 具有两个轴以上的张量没有特殊的数学名称。

#  arange 创建一个行向量 x。这个行向量包含以0开始的前12个整数，它们默认创建为整数。
x = torch.arange(12)
print(x)
# 通过张量的shape属性来访问张量（沿每个轴的长度）的形状 。
print(x.shape)
# 张量中元素的总数，即形状的所有元素乘积，可以检查它的大小（size）。 因为这里在处理的是一个向量，所以它的shape与它的size相同。
print(x.numel())

# 改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数。
# 可以把张量x从形状为（12,）的行向量转换为形状为（3,4）的矩阵。
# 这个新的张量包含与转换前相同的值，但是它被看成一个3行4列的矩阵。
# 要重点说明一下，虽然张量的形状发生了改变，但其元素值并没有变。 注意，通过改变张量的形状，张量的大小不会改变。
# 我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。使用-1可以自动计算出维度
X = x.reshape(3, 4)
print(X)
# 此时 torch.Size([3, 4])
print(X.shape)

# print(torch.zeros(2, 3, 4))
#
# print(torch.ones(2,3,4))
# 创建一个形状为（3,4）的张量。 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
# print(torch.randn(3, 4))


# [-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素
print(X[-1])
print(X[1:3])



# 数据预处理
import os

os.makedirs(os.path.join('../..', 'data'), exist_ok=True)
data_file = os.path.join('../..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)


# 处理缺失值
# 插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
# 通过位置索引iloc，我们将data分成inputs和outputs， 其中前者为data的前两列，而后者为data的最后一列。 对于inputs中缺少的数值，我们用同一列的均值替换“NaN”项。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y



