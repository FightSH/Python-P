import numpy as np
import torch
import torch.nn as nn
from numexpr.necompiler import double
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For Progress Bar
from tqdm import tqdm

# prepare dataset
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS',
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]

feature_num = len(feature_names)


class BostonDataset(Dataset):
    def __init__(self, filepath):
        original_data = np.fromfile(filepath, sep=' ',dtype=double)

        data = original_data.reshape([original_data.shape[0] // feature_num, feature_num])
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data[:, :-1])
        self.y_data = torch.from_numpy(data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



dataset = BostonDataset('work/housing.data')
testset = BostonDataset('work/test.data')
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)  # num_workers 多线程
test_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)  # num_workers 多线程



# design model using class
class My_Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 6).double()  # 输入数据x的特征是8维，x有8个特征
        self.relu = nn.ReLU().double()

        self.linear3 = torch.nn.Linear(6, 1).double()

        # # TODO: modify model's structure, be aware of dimensions.
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, 6).double(),
        #     nn.ReLU(),
        #     nn.Linear(6, 1).double()
        #
        # )

    def forward(self, x):
        x =self.linear1(x)
        x =self.relu(x)
        x =self.linear3(x)

        return x

model = My_Model(feature_num-1)
# construct loss and optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



if __name__ == '__main__':
    # training cycle forward, backward, update

    train_pbar = tqdm(range(1000), position=0, leave=True)



    for epoch in range(1000):
        model.train()  # Set your model to train mode.
        loss_record = []
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            optimizer.zero_grad()
            inputs, labels = data
            # print(inputs.shape)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss_record.append(loss.detach().item())
            # print(epoch, i, loss.item())
            # optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        mean_train_loss = sum(loss_record) / len(loss_record)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []

        for x, y in test_loader:
            # x, y = x.to(device), y.to(device)
            with torch.no_grad():
                # 注意，我们只在train模式下才会计算梯度，在validation和test模式下都需要通过torch.no_grad()把torch调整到非梯度模式。
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.detach().item())

        mean_valid_loss = sum(loss_record) / len(loss_record)



