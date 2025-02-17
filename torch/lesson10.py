import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# prepare dataset

batch_size = 32

# 存储训练轮数以及对应的accuracy用于绘图
epoch_list = []
acc_list = []


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class

# 输出尺寸 = (输入尺寸 - 卷积核尺寸 + 2× 填充) / 步长 + 1
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        x = self.fc(x)

        return x

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.layers = nn.Sequential(
#             # 黑白图像，输入通道为1，设置10个滤波器
#             nn.Conv2d(1, 10, kernel_size=5),
#             nn.ReLU(),
#             # pooling 窗口为 2
#             nn.MaxPool2d(2),
#             # 输入通道为10，设置20个滤波器
#             nn.Conv2d(10, 20, kernel_size=5),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc = nn.Linear(320, 10)
#         self.fc = torch.nn.Linear(320, 10)
#
#     def forward(self, x):
#         x = self.layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))
    acc_list.append(correct / total)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
        epoch_list.append(epoch)


# loss曲线绘制，x轴是epoch，y轴是loss值
plt.plot(epoch_list, acc_list)
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.show()
