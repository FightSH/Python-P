# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
import tempfile
import shutil
from torch.utils.tensorboard import SummaryWriter

# 为神经网络的训练提供一致的随机种子，确保训练结果的可复现性。它的输入 seed 是一个整数，我们可以在 2.7. 中的config里设置它。
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 可以根据我们给定的验证集比例（valid_ratio）将原始的训练集随机划分为训练集和验证集，以供训练过程使用。
# 它需要 3 个输入参数：未分割的训练集（data_set），验证集比例（valid_ratio）和随机种子数（seed）。
# 加入人工设置的随机种子数的目的也是为了使得分割方式在每一次训练的尝试中保持一致，使模型的训练结果有更强的可比性。
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

# 模型测试所用函数。我们需要对其输入测试集（test_loader），训练好的模型（model）和跑模型的设备（device）。其输出值为我们训练好的模型对测试集的预测结果。
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):  # tqmd可作为for循环迭代器使用，同时也提供进度条服务
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu()) # detach()函数从原tensor中剥离出一个新的相等tensor，并将新tensor放入cpu。
    preds = torch.cat(preds, dim=0).numpy() # 将preds列表拼接成tensor，再转化为np array。
    return preds

#
class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

# 实例代码有全连接神经网络，使用ReLU函数作为线性层之间的激活函数
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x

# 选取特征
# def select_feat(train_data, valid_data, test_data, select_all=True):
#     '''Selects useful features to perform regression'''
#     # 将最后一列作为目标变量
#     y_train, y_valid = train_data[:, -1], valid_data[:, -1]
#     # 将其余列作为特征
#     raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data  # 标签y取最后一列，特征x取前面的所有列。
#
#     if select_all:
#         # 如果选择所有特征，生成所有特征列的索引列表
#         feat_idx = list(range(raw_x_train.shape[1]))
#     else:
#         # 如果不选择所有特征，手动指定特征列的索引
#         feat_idx = list(range(1, 37))
#         for i in range(5):
#             # 通过 for 循环，依次添加从 37 开始的每 16 个连续的 13 个索引，共 5 次
#             # 主要是排除掉心理因素
#             feat_idx += list(range(37 + i * 16, 37 + i * 16 + 13))
#         # feat_idx = [0, 1, 2, 3, 4]  # TODO: Select suitable feature columns.
#     # 返回选定的特征数据和对应的目标变量
#     print(f"""feat_idx size: {feat_idx} """)
#
#     # return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid
#     idx_ = raw_x_train[:, feat_idx]
#     feat_idx_ = raw_x_valid[:, feat_idx]
#     test_feat_idx_ = raw_x_test[:, feat_idx]
#     return idx_, feat_idx_, test_feat_idx_, y_train, y_valid


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]  # 此处train_data和valid_data为未分离特征值和标签的数据。
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data  # 标签y取最后一列，特征x取前面的所有列。

    if select_all:  # 当选取所有特征作为训练数据时。
        feat_idx = list(range(raw_x_train.shape[1]))  # raw_x_train.shape=[条目数, 特征数]，取特征数的维度数作为特征总数。
    else:  # 当选取部分特征(用户自定义)作为训练数据时。
        # TODO: Select suitable feature colums.
        feat_idx = list(range(1, 37))
        for i in range(5):
            feat_idx += list(range(37 + i * 16, 37 + i * 16 + 13))
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

def trainer(train_loader, valid_loader, model, config, device):
    """训练模型并进行验证。

    参数:
    train_loader: 训练数据的 DataLoader。
    valid_loader: 验证数据的 DataLoader。
    model: 要训练的模型。
    config: 配置字典，包含训练参数。
    device: 模型和数据所在的设备。
    """
    # 'mean'，会以tensor(1)的形式输出各维度MSE的平均值
    # 'sum'，则会以tensor(1)的形式输出各维度MSE的总和[4]
    # 'none'，则会以tensor的形式输出y矩阵各维度的MSE
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    # TensorBoard 写入器
    writer = SummaryWriter()  # Writer of tensoboard.
    # 创建保存模型的目录（如果不存在）
    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.
    # 初始化训练参数
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        # 设置模型为训练模式
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            # 在 tqdm 进度条上显示当前轮次编号和损失。
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        # 计算平均训练损失
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        # 验证
        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
        # 计算平均验证损失
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        # 保存最佳模型（基于验证损失）
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


def trainer2(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.
    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optiminzer(weight decay...) or implement by yourself).
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    writer = SummaryWriter()  # Writer of tensorboard.

    if not os.path.isdir('./models'):  # 此处利用os.path.isdir()函数判断模型存储路径是否存在，以避免os.mksir()函数出错。
        os.mkdir('./models')  # Create directory of saving models.

    # 此处定义模型训练的总轮数(n_epochs)以及一系列用于计数的变量(best_loss, step, early_stop_count)。
    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    # best_loss = math.inf 返回浮点数正无穷(+∞)

    train_pbar = tqdm(range(n_epochs), position=0, leave=True)

    for epoch in train_pbar:
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        # train_pbar = tqdm(train_loader, position=0, leave=True)
        # position=0可以防止多行进度条的情况（？说实话，还是不够清楚理解）。

        for x, y in train_loader:
            optimizer.zero_grad()  # Start gradient to zero.
            x, y = x.to(device), y.to(device)  # data.to()函数将数据移至指定设备(CPU/GPU)。
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient (backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            # 使用PyTorch时，特别需要清楚每个变量的数据类型并在需要时进行变量的拷贝和转换(尤其是在遇到tensor数据类型时)。

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                # 注意，我们只在train模式下才会计算梯度，在validation和test模式下都需要通过torch.no_grad()把torch调整到非梯度模式。
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.detach().item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        # print(f'Epoch [{epoch+1}/{n_epochs}]: Train_loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:  # 将最佳loss值更新到best_loss变量中。
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # 保存当前步骤的最佳model。
            # print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1  # 记录模型未能优化的次数，为模型收敛中断训练提供参考。

        # Display current epoch number and loss on tqdm progress bar.
        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        train_pbar.set_postfix({'Best loss': '{0:1.5f}'.format(best_loss)})

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return None


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,      # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,   # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

# Set seed for reproducibility
same_seed(config['seed'])


# train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
# test_data size: 1078 x 117 (without last day's positive rate)
# 加载数据
train_data, test_data = pd.read_csv('./data/covid.train.csv').values, pd.read_csv('./data/covid.test.csv').values
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# Print out the data size.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

# Select features
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

# Print out the number of features.
print(f'number of features: {x_train.shape[1]}')

# 创建数据集
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# Pytorch data loader loads pytorch dataset into batches.
# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)


# 初始化模型
model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
# 开始训练
trainer2(train_loader, valid_loader, model, config, device)


tb_info_dir = os.path.join(tempfile.gettempdir(), '.tensorboard-info')  # 获取tensorboard临时文件地址
shutil.rmtree(tb_info_dir)  # 递归删除该临时文件所在目录



def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
# 加载最佳模型并进行预测
model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
# 保存预测结果
save_pred(preds, 'pred.csv')

