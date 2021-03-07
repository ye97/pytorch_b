import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        #super(DiabetesDataset, self).__init__()
        xy=np.loadtxt(filepath,dtype=np.float32,delimiter=",")
        self.len=xy.shape[0]
        self.x=torch.from_numpy(xy[:,:-1])
        self.y=torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, item):
        return self.x[item],self.y[item]

    def __len__(self):
        return self.len



dataset = DiabetesDataset('diabetes.csv')
#trainloader 第一个参数dataset类对象，batchsize会将dataset划分多批，多批*batchsize>dataset.len
#shuffle表示乱序提取
#newworker表示线程数；注意多线程一定要包括在main函数中，否则就会报错

train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2) #num_workers 多线程


# design model using class


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

# construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.3)

#所有训练和更新代码都应该使用main函数包裹起来
if __name__ == '__main__':

    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # train_loader 是先shuffle后mini_batch
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
