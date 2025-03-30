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


# 数据集合划分
def train_valid_split(data_set, valid_ratio, seed):
    valid_data_size = (int) (len(data_set) * valid_ratio)
    train_data_size = len(data_set) - valid_data_size
    train_data, valid_data = random_split(data_set, [train_data_size, valid_data_size],
                                          generator=torch.Generator().manual_seed(seed))
    # 以np数组形式返回
    return np.array(train_data), np.array(valid_data)


# 选择特征
# 选取特征
def select_feat(train_data, valid_data, test_data, select_all=True):
    # 选最后一列 是label
    y_train = train_data[:, -1]
    y_valid = valid_data[:, -1]

    # 选择除最后一列的所有元素
    raw_x_train = train_data[:, :-1]

    raw_x_valid = valid_data[:, :-1]
    # 测试集没最后的label
    raw_x_test = test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = list(range(1, 37))
        for i in range(5):
            feat_idx += list(range(37 + i * 16, 37 + i * 16 + 13))
    print(feat_idx)
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


# 数据集类
class COVID19Dataset(Dataset):
    #  y: Targets, if none, do prediction.
    def __init__(self, features, targets=None):
        if targets is None:
            self.targets = targets
        else:
            self.targets = torch.FloatTensor(targets)

        self.features = torch.FloatTensor(features)

    def __getitem__(self, idx):
        if self.targets is None:
            return self.features[idx]
        else:
            # 做训练，需要特征和label
            return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)


# 神经网络模型
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        # 压缩张量的第一个维度
        x = x.squeeze(1)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,  # Your seed number, you can pick your lucky number. :)
    'select_all': False,  # Whether to use all features.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 3000,  # Number of epochs.
    'batch_size': 128,
    'learning_rate': 1e-5,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}


# for epoch in range(n_epochs):                             iterate n_epochs
#     model.train()                                         set model to train mode
#     for x,y in tr_set:                                    iterate through the dataloader
#         optimizer.zero_grad()                             set gradient to zero
#         x,y=x.to(device),y.to(device)                     move data ti device
#         pred = model(x)                                   forward pass
#         loss = criterion(pred,y)                          compute loss
#         loss.backward()                                   compute gradient
#         optimizer.step()                                  update model with optimizer
#     model.eval()
#     total_loss = 0
#     for(x,y) in dv_set:
#         x,y = x.to(device),y.to(device)
#         with torch.no_grad():                             disable gradient calculation
#               pred = model(x)
#               loss = criterion(pred, y)                   compute loss
#         total_loss += loss.cpu.item()*len(x)              accumulate loss
#         avg_loss = total_loss/len(dv_set.dataset)         avg loss

def trainer(train_loader, valid_loader, model, config, device):
    # 均方误差
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    writer = SummaryWriter()  # 数据可视化
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs = config['n_epochs']
    best_loss = math.inf  # 无穷大
    step = 0
    early_stop_count = 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        # 进度条就是封装了dataloader
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            step += 1
            # 记录损失值
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        # 评价loss
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # valid
        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train_loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...', format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
        train_pbar.set_postfix({'Best loss': '{0:1.5f}'.format(best_loss)})

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return None

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

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

def print_model_info(model, print_param_values=False):
    """
    Prints model architecture and parameters

    Args:
        model: PyTorch model
        print_param_values: If True, prints actual parameter values (can be verbose)
    """
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE:")
    print("="*50)
    print(model)

    print("\n" + "="*50)
    print("MODEL PARAMETERS:")
    print("="*50)

    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        print(f"{name}: shape={tuple(param.shape)}, parameters={param_count}")
        if print_param_values:
            print(f"Values: {param.data}")

    print("\n" + "="*50)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("="*50 + "\n")

# 设置种子
same_seed(config['seed'])
# 读取数据
train_data, test_data = pd.read_csv('./data/covid.train.csv').values, pd.read_csv('./data/covid.test.csv').values

# 划分数据
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
print(f'number of features: {x_train.shape[1]}')

# 构造数据集
train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                            COVID19Dataset(x_valid, y_valid), \
                                            COVID19Dataset(x_test)

# 准备DataLoader
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
# Print model architecture and parameters before training
print_model_info(model)

# 开始训练
trainer(train_loader, valid_loader, model, config, device)



model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
# Print model architecture and parameters after loading trained model
print_model_info(model)
preds = predict(test_loader, model, device)
# 保存预测结果
save_pred(preds, 'pred.csv')



