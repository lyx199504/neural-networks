#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/4/1 13:27
# @Author : LYX-夜光

import paddle
import pandas as pd
from paddle import nn

from paddle.io import Dataset, DataLoader


class GetDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.feat = ['weight', 'height']
        self.data = data
        self.mean = data.mean()
        self.std = data.std()

    def __getitem__(self, index):
        data = []
        for feat in self.feat:
            data.append((self.data.loc[index, feat] - self.mean[feat])/self.std[feat])
        data = paddle.to_tensor(data)
        print(data)
        if self.data.shape[1] == 3:
            label = paddle.to_tensor(self.data.loc[index, 'label'], dtype='float32')
        else:
            label = paddle.to_tensor(-1, dtype='float32')
        return data, label

    def __len__(self):
        return len(self.data)

class ConNet(paddle.nn.Layer):
    def __init__(self):
        super(ConNet, self).__init__()
        self.hidden_layer = nn.Linear(in_features=2, out_features=2)
        self.out_layer = nn.Linear(in_features=2, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.hidden_layer(x)
        y = self.sigmoid(y)
        y = self.out_layer(y)
        y = self.sigmoid(y)
        return y


if __name__ == "__main__":
    trainset = pd.DataFrame({'weight': [133., 160, 152, 120], 'height': [65., 72, 70, 60], 'label': [0, 1, 1, 0]})
    trainset = GetDataset(trainset)
    trainset.__getitem__(0)
    exit()
    train_loader = DataLoader(trainset, batch_size=4)

    lr = 0.5
    epochs = 2000
    loss_fn = nn.MSELoss()

    model = ConNet()
    model.train()  # 训练模式开启
    optimizer = paddle.optimizer.SGD(parameters=model.parameters(), learning_rate=lr)  # 优化器

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            X, y = data
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            if epoch % 100 == 99:
                print("epoch: %d/%d - loss is: %.6f" % (epoch+1, epochs, float(loss)))

            optimizer.clear_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

    testset = pd.DataFrame({'weight': [128, 155.], 'height': [63., 68.]})
    testset = GetDataset(testset)
    test_loader = DataLoader(testset, batch_size=1)

    model.eval()
    for i, data in enumerate(test_loader, 0):
        X = data[0]
        y_pred = model(X)
        print(float(y_pred))