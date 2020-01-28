#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/19 10:56
# @Author : LYX-夜光

import torch
import numpy as np
from collections import OrderedDict

# 神经网络
def networks(x, y):
    # 数据转化为张量
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()

    input_data = x.shape[1]  # 输入（特征）个数
    hidden_layer = x.shape[1]  # 隐藏层个数
    output_data = y.shape[1]  # 输出个数

    # 神经网络模型
    model = torch.nn.Sequential(OrderedDict([
        ("Line1", torch.nn.Linear(input_data, hidden_layer)),  # 输入层至隐藏层的线性变化
        ("Sigm1", torch.nn.Sigmoid()),  # 激活函数
        ("Line2", torch.nn.Linear(hidden_layer, output_data)),  # 隐藏层至输出层的线性变化
        ("Sigm2", torch.nn.Sigmoid()),  # 激活函数
    ]))

    learningRate = 0.5  # 训练参数
    epochs = 2000  # 迭代次数
    loss_fn = torch.nn.MSELoss()  # 损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  # 优化器

    for epoch in range(epochs):  # 每一次迭代
        yPred = model(x)
        loss = loss_fn(yPred, y)
        if epoch % 100 == 0:
            print("Epoch %d loss: %.3f" % (epoch, float(loss)))
        optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）
        loss.backward()  # 梯度计算
        optimizer.step()  # 优化更新权值
    return model

if __name__ == "__main__":
    data = np.array([[133, 65],  # 每一行是体重（磅）和身高（英寸）
                     [160, 72],
                     [152, 70],
                     [120, 60]])
    dataMean, dataStd = data.mean(), data.std()
    newData = (data - dataMean) / dataStd  # 标准化
    yTrues = np.array([[0, 1, 1, 0]]).T  # 1男性 0女性

    model = networks(newData, yTrues)

    # 测试样例
    Alice = np.array([128, 63])
    dataAlice = torch.tensor(((Alice - dataMean) / dataStd)).float()
    gender = model.forward(dataAlice)
    print("Alice: %f" % gender)

    Frank = np.array([155, 68])
    dataFrank = torch.tensor(((Frank - dataMean) / dataStd)).float()
    gender = model.forward(dataFrank)
    print("Frank: %f" % gender)