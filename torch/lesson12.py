import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
Cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)  # 初始化，构建RNNCell
dataset = torch.randn(seq_len, batch_size, input_size)  # 设置dataset的维度
print(dataset)
print(dataset.shape)
hidden = torch.zeros(batch_size, hidden_size)  # 隐层的维度：batch_size*hidden_size，先把h0置为0向量
for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size:', input.shape)
    hidden = Cell(input, hidden)
    print('Outputs size:', hidden.shape)
    print(hidden)
