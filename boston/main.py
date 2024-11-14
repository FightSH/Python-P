import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
# xy = np.loadtxt('./data/diabetes.csv', delimiter=',', dtype=np.float32)
data = np.fromfile('./work/housing.data', sep=' ')
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])



offset = int(data.shape[0] * 0.8)
training_data = data[:offset]

# 计算训练集的最大值，最小值
maximums, minimums = training_data.max(axis=0), \
    training_data.min(axis=0)

# 对数据进行归一化处理
for i in range(feature_num):
    data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])

training_data = data[:offset]
test_data = data[offset:]

x_data = torch.from_numpy(training_data[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
y_data = torch.from_numpy(training_data[:, [-1]])  # [-1] 最后得到的是个矩阵

# Print out the number of features.
print(f'number of features: {x_data.shape[1]}')
print(f'number of features: {y_data.shape[1]}')
# design model using class


class Model(torch.nn.Module):
    def __init__(self,input_dim):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 6)
        # 输入数据x的特征是8维，x有8个特征
        self.ReLU(),
        self.linear2 = torch.nn.Linear(6, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Model(feature_num)

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_list = []
loss_list = []
# training cycle forward, backward, update
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()

    optimizer.step()# update 参数，即更新w和b的值

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()






