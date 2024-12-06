# 使用RNNCell
import torch

# 参数
input_size = 4
hidden_size = 4
batch_size = 1
# 准备数据
idx2char = ['e', 'h', 'l', 'o']  # 为了后面可以根据索引把字母取出来
x_data = [1, 0, 2, 3, 3]  # hello中各个字符的下标
y_data = [3, 1, 2, 3, 2]  # ohlol中各个字符的下标
# 独热向量
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]  # (seqLen, inputSize)
# reshape the inputs to (seqlen,batchSize,inputSize)
inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# reshape the labels to (seqlen,1)
labels = torch.LongTensor(y_data).view(-1, 1)
# torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
print(inputs.shape, labels.shape)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        # 对参数进行初始化
        self.batch_size = batch_size  # 仅构造h0时需要batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)

    def forward(self, inputs, hidden):
        hidden = self.rnncell(inputs, hidden)  # 输入和隐层转换为下一个隐层 ht = rnncell(xt,ht-1)
        # shape of inputs:(batchSize, inputSize),shape of hidden:(batchSize, hiddenSize),
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)  # 提供初始的隐层，生成全0的h0


net = Model(input_size, hidden_size, batch_size)
# 损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)  # 使用Adam优化器，改进的随机梯度下降优化器进行优化

for epoch in range(15):
    loss = 0
    optimizer.zero_grad()  # 优化器梯度归0
    hidden = net.init_hidden()  # 每一轮的第一步先初始化hidden,即先计算h0
    print('Predicted string:', end='')
    # shape of inputs:(seqlen序列长度,batchSize,inputSize)  shape of input:(batchSize,inputSize)
    # shape of labeis:(seqsize序列长度,1)  shape of labei:(1)
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 注意交叉熵在计算loss的时候维度关系，这里的hidden是([1, 4]), label是 ([1])
        loss += criterion(hidden, label)  # 不用loss.item,所有的和才是最终的损失
        _, idx = hidden.max(dim=1)  # hidden.max()函数找出hidden里的最大值  _, idx最大值的下标
        print(idx2char[idx.item()], end='')

    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))