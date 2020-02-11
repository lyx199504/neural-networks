#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/2/8 13:33
# @Author : LYX-夜光

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 获取数据集类
class FaceDataset(Dataset):
    def __init__(self, root, transform=None, numPerLabel=10):
        self.images = [os.path.join(root, image) for image in os.listdir(root)]
        self.transform = transform
        if not self.transform:
            self.transform = transforms.Compose([  # 初始化图像
                transforms.CenterCrop(224),  # 将图像中心裁剪为224*224
                transforms.ToTensor(),  # 将图像转化为张量并且归一化
                transforms.Normalize([.5], [.5]),  # 将张量标准化
            ])
        self.numPerLabel = numPerLabel

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imagePath = self.images[index]
        data = Image.open(imagePath)
        if self.transform:
            data = self.transform(data)
        label = int(index / self.numPerLabel)
        return data, label

# 获取训练集
def getTrainDataSet(model, device, batch_size=10, transform=None, numPerLabel=20):
    rawDataset = FaceDataset('./data/train', transform=transform, numPerLabel=numPerLabel)
    # 选择三元组数据
    image = torch.tensor([rawDataset[0][0].tolist()]).to(device)
    with torch.no_grad():
        anchor = model(image).cpu()  # anchor图像向量
    # 创建三元hard数据集
    distList = []  # negative与anchor的距离
    for i in range(numPerLabel, len(rawDataset)):
        image = torch.tensor([rawDataset[i][0].tolist()]).to(device)
        with torch.no_grad():
            negative = model(image).cpu()  # negative图像向量
        dist = torch.dist(negative, anchor)  # 欧式距离
        distList.append([i, dist])
    distList = sorted(distList, key=lambda x: x[-1])  # 距离从小到大排序
    negativeNum = 200  # 取出与anchor距离最小的negativeNum个下标（取200个是因为电脑配置有限）
    negativeIndexList = list(map(lambda x: x[0], distList[:negativeNum]))
    train_dataSet = []  # 三元hard数据集
    for positive_index in range(1, numPerLabel):  # positive
        for negative_index in negativeIndexList:  # negative
            train_dataSet.append([rawDataset[0][0], rawDataset[positive_index][0], rawDataset[negative_index][0]])
            # # 检查选取的三元数据
            # print(rawDataset[0][1], rawDataset[positive_index][1], rawDataset[negative_index][1])
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

# 获取测试集
def getTestDataSet(batch_size=4, transform=None, numPerLabel=10):
    test_dataSet = FaceDataset('./data/test', transform=transform, numPerLabel=numPerLabel)
    test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=False, num_workers=2)
    return test_loader, test_dataSet[0]

# 获取验证集（没有验证集，用训练集）
def getVerifyDataSet(batch_size=4, transform=None, numPerLabel=20):
    verify_dataSet = FaceDataset('./data/train', transform=transform, numPerLabel=numPerLabel)
    verify_loader = DataLoader(verify_dataSet, batch_size=batch_size, shuffle=False, num_workers=2)
    return verify_loader, verify_dataSet[0]
