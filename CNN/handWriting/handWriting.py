#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/26 22:29
# @Author : LYX-夜光

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),  # 卷积，将1张28*28的图像卷积成8个26*26的图像
            torch.nn.BatchNorm2d(8),  # 标准化（下同）
            torch.nn.ReLU(inplace=True),  # 激活函数，将小于0的值变为0（下同）
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3),  # 将8张26*26的图像卷积成16个24*24的图像
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，将24*24的图像池化成12*12的图像
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 24, kernel_size=3),  # 将16张12*12的图像卷积成24个10*10的图像
            torch.nn.BatchNorm2d(24),
            torch.nn.ReLU(inplace=True),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(24, 32, kernel_size=3),  # 将24张10*10的图像卷积成32个8*8的图像
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 将8*8的图像池化成4*4的图像
        )

        self.function = torch.nn.Sequential(  # 全连接层
            torch.nn.Linear(32 * 4 * 4, 128),  # 输入层-隐藏层
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 10)  # 隐藏层-输出层，输出10个分类
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 将128*4*4的图像平铺成一维
        x = self.function(x)
        return x

def train(datas, learning_rate, device):
    model = CNN().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    epoch = 0
    for data in datas:
        img, label = data[0].to(device), data[1].to(device)
        out = model(img)
        loss = loss_fn(out, label)
        epoch += 1
        if epoch % 50 == 0:
            print("Epoch %d loss: %.3f" % (epoch, float(loss)))
        optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）
        loss.backward()  # 梯度计算
        optimizer.step()  # 优化更新权值
    return model

def test(model, datas, device):
    model.eval()  # 使model不再改变权值
    correctNum = 0.0
    for data in datas:
        img, label = data[0].to(device), data[1].to(device)
        out = model(img)
        _, pred = torch.max(out, 1)  # 获得10个分类中最大值的下标，下标为预测值
        correctNum += (pred == label).sum().item()
    return correctNum

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 在GPU或CPU运行
    # transforms.ToTensor初始化数据为张量，并将数据归一化为区间为[0,1]的数值
    # transforms.Normalize数据标准化，即(数据-均值)/(标准差)，第一个参数为比均值，第二个参数为标准差
    # transforms.Compose将上面两个过程整合在一起
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    train_dataSet = datasets.MNIST("./data", train=True, transform=transform)
    test_dataSet = datasets.MNIST("./data", train=False, transform=transform)
    # 将数据分成train_loader组，每组batch_size个数据
    batch_size = 64
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=False)
    path = "model.pth"
    if os.path.exists(path):  # 模型存在则加载模型
        model = torch.load(path).to(device)
    else:  # 不存在则训练模型
        model = train(train_loader, 0.02, device=device)
        torch.save(model, path)
    correctNum = test(model, test_loader, device=device)
    print("Test correct rate: %.3f" % (correctNum / len(test_dataSet)))