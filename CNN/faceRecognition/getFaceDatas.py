#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/2/8 13:33
# @Author : LYX-夜光

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random

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

# 获取三元组数据集
class TripleDataset(Dataset):
    def __init__(self, rawDataset, numPerLabel=20):
        self.dataset = []  # 生成三元组数据集
        anchor_index = 0  # 选择第1张作为主图像
        for i in range(1, numPerLabel):  # 剩下19张作为positive
            positive_index = i
            randList = list(range(numPerLabel, len(rawDataset)))
            random.shuffle(randList)
            negativeNum = 250  # 随机取positiveNum张作为negative
            for j in randList[:negativeNum]:
                negative_index = j
                # # 检查选取的三元组
                # print(rawDataset[anchor_index][1], rawDataset[positive_index][1], rawDataset[negative_index][1])
                self.dataset.append([rawDataset[anchor_index][0], rawDataset[positive_index][0], rawDataset[negative_index][0]])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index][0], self.dataset[index][1], self.dataset[index][2]

# 获取训练集
def getTrainDataSet(batch_size=10, transform=None, numPerLabel=20):
    rawDataset = FaceDataset('./data/train', transform=transform, numPerLabel=numPerLabel)
    # 选择三元组数据
    train_dataSet = TripleDataset(rawDataset, numPerLabel)
    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

# 获取测试集
def getTestDataSet(batch_size=4, transform=None, numPerLabel=10):
    test_dataSet = FaceDataset('./data/test', transform=transform, numPerLabel=numPerLabel)
    test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=False, num_workers=2)
    anchors = DataLoader([test_dataSet[0]], batch_size=1, shuffle=False)
    return test_loader, anchors
