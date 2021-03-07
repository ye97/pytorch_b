from pathlib import Path
import requests
import pickle
import gzip
FILENAME = Path("D:\pytorch_b\mnist.pkl.gz")
with gzip.open((FILENAME).as_posix(),"rb") as f:
    ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f,encoding="latin-1")
from matplotlib import pyplot
import numpy as np
pyplot.imshow(x_train[0].reshape((28,28)),cmap="gray")

import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

import torch.nn.functional as F

loss_func = F.cross_entropy #分类问题 交叉熵损失函数

def model(xb):
    return xb.mm(weights)+bias
bs = 64   # batch size
xb = x_train[0:bs]  # a mini-batch from x
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True)
bias = torch.zeros(10, requires_grad=True)  # 偏置参数

print(loss_func(model(xb), yb))

from torch import nn


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784,128) #隐层1
        self.hidden2 = nn.Linear(128,256) #隐层2
        self.out = nn.Linear(256,10) #输出
    def forward(self,x):
        x = F.relu(self.hidden1(x)) #激活函数 结果传送到relu中 得到第一层处理结果
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Mnist_NN()
print(net)

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#封装成dataset格式
train_ds = TensorDataset(x_train,y_train)
train_dl = DataLoader(train_ds,batch_size=bs,shuffle=True) #shuffle洗牌

valid_ds = TensorDataset(x_valid,y_valid)
valid_dl = DataLoader(valid_ds,batch_size=bs*2)


#用dataloader一个batch一个batch的读取数据
def get_data(train_ds,valid_ds,bs):
    return(
        DataLoader(train_ds,batch_size=bs,shuffle=True),
        DataLoader(valid_ds,batch_size=bs,shuffle=True)
    )

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

from torch import optim
def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)

import numpy as np
#steps=要迭代多少次
#model模型
#loss_func损失函数
#opt优化器
#train_dl训练数据
#valid_dl验证数据
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:'+str(step), '验证集损失：'+str(val_loss))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, valid_dl)