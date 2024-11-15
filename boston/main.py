import numpy as np
import torch
import torch.nn as nn
from numexpr.necompiler import double
from torch.utils.data import Dataset
from torch.utils.data import DataLoader





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
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=2)  # num_workers 多线程



# design model using class
class My_Model(torch.nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 6).double(),
            nn.ReLU(),
            nn.Linear(6, 1).double()

        )

    def forward(self, x):
        x = self.layers(x)
        return x

model = My_Model(feature_num-1)
# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


if __name__ == '__main__':
    # training cycle forward, backward, update
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            print(inputs.shape)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()









