# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import random
import os
import csv
import gc


# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter



def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 读取pt文件
def load_feat(path):
    feat = torch.load(path)
    # 获取的是张量
    # 读取pt文件，获取的张量维度为[T,39]，即可能有多个帧，每个帧都有39维
    return feat

'''
x=X[2][-1],n=1 注意x[2] 
right = X[2][-1].repeat(1,1) =  X[2][-1]  这里的-1相当于选了最后一列的数据  第2组，最后一帧
left = X[2][1:] 相当与选择了X[2]除第一列的所有列
cat((X[2][-1],X[2][1:]), dim=0) 沿着第0维拼接，即横着拼起来

x=X[0],n=-1
left = X[0][0].repeat(1,1) = X[0][0] 选了第0组，第一帧
right = X[0][]
'''
def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


# 拼接的帧
'''
x  pt[T,39]
concat_n : 拼接的frames数目
'''
def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len = x.size(0) # 帧的个数
    feature_dim =  x.size(1) # 每一个 frame 的维度 39

    # 沿着第一维重复N次
    '''
        eg 原来是这样  0 1 2 3 4 5 6 7 8 9
        变成这样
                     0 1 2 3 4 5 6 7 8 9
                     0 1 2 3 4 5 6 7 8 9
                     ......
                     ......
                     0 1 2 3 4 5 6 7 8 9
       见图帧.png             
    '''
    x = x.repeat(1, concat_n)
    '''
        假设 concat_n = 3，seq_len = T，feature_dim = 39
        view[seq_len, concat_n, feature_dim] = [T,3,39]
        permute[concat_n,seq_len,feature_dim] = [3,T,39]
        [组号，帧序号，帧内数据序号]
    
        view重构张量的维度或形状 view[seq_len, concat_n, feature_dim]
        permute则是调换维度顺序 permute[concat_n,seq_len,feature_dim]
    '''
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim

    # 取到中间的帧，即待预测的帧
    mid = (concat_n // 2)
    '''
    mid = 3//2 = 1
    x[1+1,:] = x[2] = shift(x[1+1],1) = shift(x[2],1)
    x[1-1,:] = x[0] = shift(x[1-1],-1) = shift(x[0],-1)
    '''
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)
    #     处理后见拼接后的帧
    # 三维再处理成二维 把所有的帧拼成长长的向量  (11, 3*39 )
    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


# 预处理数据
def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    # 音素固定41类
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {} # 索引字典  测试数据建立字典
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          # 第0个元素就是文件名
          label_dict[line[0]] = [int(p) for p in line[1:]]
          '''[int(p) for p in line[1:]] = 下面的
                temp = []
                for p in line[1:]
                    temp.append(int(p))
          '''

    if split == 'train' or split == 'val':
        # split training and validation data  文件中划分，毕竟txt是文件名
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio) # 训练集的百分比
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        # 测试数据不划分
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')
    # 删除换行符
    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    #
    max_len = 3000000
    # 空张量
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      # label值
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        # 读pt文件
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        # frame拼接
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          #   名字为key，拿到label值
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len


    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.35)
        )

    def forward(self, x):
        x = self.block(x)
        return x

# 模块化的优点
class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),  # 输入层
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)], # 隐含层
            nn.Linear(hidden_dim, output_dim)   # 输出层
        )

    def forward(self, x):
        x = self.fc(x)
        return x


## Hyper-parameters

# data prarameters
concat_nframes = 11              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.95               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 0                        # random seed
batch_size = 2048                # batch size
num_epoch = 10                   # the number of training epoch
learning_rate = 0.0001          # learning rate
model_path = './model.ckpt'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 3               # the number of hidden layers
hidden_dim = 256




## prepare dataset and model

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')



# fix random seed
same_seed(seed)

# create model, define a loss function, and optimizer
model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=8,T_mult=2,eta_min = learning_rate/2)

# training
best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 得到可能性最高的类别标号   train_pred value 所在的index
        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        # 看有多少帧是统计正确了
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()

    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, dim = 1)
                val_acc += (
                            val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')


del train_loader, val_loader
gc.collect()


## test
# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))