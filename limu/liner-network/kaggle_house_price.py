import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l


train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')

print(train_data.shape)
print(test_data.shape)
# 使用 train_data.iloc[:, 1:-1] 选择 train_data 中从第2列到倒数第2列的所有行
# 使用 test_data.iloc[:, 1:] 选择 test_data 中从第2列到最后列的所有行
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))


# 数据需要预处理
# 首先需要将缺失的值替换为将所有缺失的值替换为相应特征的平均值
# 并且为了将所有特征放在一个共同的尺度上， 我们通过将特征重新缩放到零均值和单位方差来标准化数据
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

