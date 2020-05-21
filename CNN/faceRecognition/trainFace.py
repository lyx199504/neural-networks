#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/2/8 13:36
# @Author : LYX-夜光

import os
import torch
from CNN.faceRecognition import getFaceDatas, faceNet

# 训练
def train(model, datas, device, learning_rate=0.0002):
    model.train()  # 训练模式开启
    triple_loss_fn = torch.nn.TripletMarginLoss(margin=20.0, p=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for i, data in enumerate(datas, 0):
        anchor, positive, negative = data[0].to(device), data[1].to(device), data[2].to(device)
        anchor_out, positive_out, negative_out = model(anchor), model(positive), model(negative)
        loss = triple_loss_fn(anchor_out, positive_out, negative_out)
        if i % 10 == 9:
            print("第%d次训练的三元损失值: %.6f" % (i+1, float(loss)))
        optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）
        loss.backward()  # 梯度计算
        optimizer.step()  # 优化更新权值
    return model

if __name__ == "__main__":
    '''
    使用facenet模型训练，想再训练的话可以运行该文件，由于设备比较差，就只让代码识别第一个人
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir = "./model"
    try:
        fileName = sorted(os.listdir(dir), reverse=True)[0]  # 尝试读取模型
        model_in = dir + '/' + fileName  # 输入模型的文件名
        model_out = dir + "/model_%03d.pth" % (int(fileName.split('.')[0].split('_')[1])+1)  # 输出模型的文件名
    except:
        model_in = None
        model_out = dir + "/model_001.pth"
    print("输入模型：", model_in, "输出模型：", model_out)
    if model_in:  # 模型存在，则加载模型
        model = torch.load(model_in).to(device)
    else:  # 模型不存在，则选择预训练模型
        model = faceNet.FaceNet().to(device)
    print("开始加载数据。。。")
    train_loader = getFaceDatas.getTrainDataSet(model, device)
    print("开始训练模型。。。")
    model = train(model, train_loader, device).cpu()
    try:
        os.mkdir(dir)
    except:
        pass
    torch.save(model, model_out)
