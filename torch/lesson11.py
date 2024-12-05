import torch

# 构造Dataloader
from torchvision import transforms  # 用于对图像进行一些处理
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F  # 使用更流行的激活函数Relu
import torch.optim as optim  # 构造优化器
import matplotlib.pyplot as plt


batch_size = 64

# 存储训练轮数以及对应的accuracy用于绘图
epoch_list = []
acc_list = []

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels,
                                     kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels,
                                     kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))  # 先做第一次卷积，然后relu
        y = self.conv2(y)  # F(x)，第二次卷积
        return F.relu(x + y)  # H(x) = F(x) + x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)  # 第一个卷积层
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)  # 第二个卷积层
        self.mp = torch.nn.MaxPool2d(2)  # 池化层

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)  # 线性层

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 带冲量的梯度下降


# 一轮训练
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # inputs输入x，target输出y
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # loss累加

        # 每300轮输出一次，减少计算成本
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 测试函数
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 让后续的代码不计算梯度
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))
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
