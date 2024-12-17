import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 超参数设置
batch_size = 64
hidden_layer_size = 2
num_classes = 10
num_epochs = 10
learning_rate = 0.001

# 数据预处理和加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 实例化模型、定义损失函数和优化器
model = RNNModel( input_size=28,hidden_size=hidden_layer_size, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28, 28)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_data, batch_target in train_loader:
                    batch_data = batch_data.view(-1, 28, 28)
                    outputs = model(batch_data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_target.size(0)
                    correct += (predicted == batch_target).sum().item()
            print(f'Epoch {epoch}, Minibatch Loss= {loss.item():.6f}, Training Accuracy= {100 * correct / total:.5f}')

# 测试循环，计算测试准确率
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 28, 28)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print(f'Testing Accuracy: {100 * correct / total:.5f}')