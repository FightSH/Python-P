import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# prepare dataset


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32,skiprows=1)
        self.len = xy.shape[0]  # shape(多少行，多少列)  shape0 矩阵行数  shape1 矩阵列数
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('./data/diabetes.csv')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)  # num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':
    # epoch  one forward pass and one backward pass of all the training examples
    # batchSize the number of training examples in one forward backward pass
    # iteration number of passes ,each pass using batchSize number of examples
    # 10000个样本  1000个batchSize  10个iteration
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            # 注意 train_loader 中的元组数据放到里data里 。相当于getItem
            inputs, labels = data
            # inputs里的数据行数是32
            # print("------")
            # print(inputs)
            # print(labels)
            # print("------")
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()