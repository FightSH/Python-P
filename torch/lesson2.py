import matplotlib.pyplot as plt
import numpy as np

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]


def forward(x):
    """
    前向传播函数，用于计算给定x和权重w的预测值。

    参数:
    x -- 输入数据

    返回:
    预测值
    """
    return x * w


def loss(x, y):
    """
    损失函数，用于计算预测值和真实值之间的平方差。

    参数:
    x -- 输入数据
    y -- 真实标签

    返回:
    当前样本的损失值
    """
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 初始化权重列表和均方误差列表
w_list = []
mse_list = []

# 遍历权重范围，计算不同权重下的损失值
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    # 遍历数据集，计算每个样本的预测值和损失值
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print("\t", x_val, y_val, y_pred_val, loss_val)

    # 计算当前权重下的均方误差，并打印
    print("MSE=", l_sum / 3)
    # 将当前权重和对应的均方误差分别添加到列表中
    w_list.append(w)
    mse_list.append(l_sum / 3)

# 绘制权重与均方误差的关系图
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()