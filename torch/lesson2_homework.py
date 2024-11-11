import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 定义输入数据
x_data = [1.0, 2.0, 3.0]
y_data = [5.0, 8.0, 11.0]

def forward(x, w, b):
    """
    定义前向传播函数

    参数:
    x -- 输入数据
    w -- 权重参数
    b -- 偏置参数

    返回:
    y_pred -- 预测值
    """
    return x * w + b

def loss(x, y, w, b):
    """
    计算损失函数

    参数:
    x -- 输入数据
    y -- 真实标签
    w -- 权重参数
    b -- 偏置参数

    返回:
    损失值
    """
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 初始化均方误差列表
mse_list = []
# 定义权重和偏置参数的范围
W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
# 生成权重和偏置参数的网格
[w, b] = np.meshgrid(W, B)

# 初始化损失值数组，与权重和偏置参数的网格具有相同的维度
loss_values = np.zeros_like(w)

# 计算损失值
for i in range(len(W)):
    for j in range(len(B)):
        l_sum = 0
        # 遍历数据集，计算每个样本的损失值并累加
        for x_val, y_val in zip(x_data, y_data):
            loss_val = loss(x_val, y_val, W[i], B[j])
            l_sum += loss_val
        # 将累加的损失值除以样本数以获得平均损失，并存储到损失值数组中
        loss_values[i, j] = l_sum / len(x_data)

# 绘制三维曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 使用权重、偏置参数和损失值绘制三维曲面图
ax.plot_surface(w, b, loss_values, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')

# 显示图形
plt.show()
