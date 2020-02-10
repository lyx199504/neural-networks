#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/2/10 0:40
# @Author : LYX-夜光

import os
import torch
from CNN.faceRecognition import getFaceDatas

# 生成图像编码
def genCode(model, datas, device):
    model.eval()  # 训练模式关闭，使model不再改变权值
    anchorList = []
    for data in datas:
        image = data[0].to(device)
        with torch.no_grad():
            out = model(image).cpu()
        anchorList.append([data[1], out])
    return anchorList

# 测试
def test(model, datas, anchors, device):
    model.eval()  # 训练模式关闭，使model不再改变权值
    anchorList = []
    for anchor in anchors:  # 获取每个主图像
        image = anchor[0].to(device)
        with torch.no_grad():
            anchorList.append(model(image).cpu())
    correctNum = 0.0
    for data in datas:
        image = data[0].to(device)
        with torch.no_grad():
            out = model(image).cpu()
        for anchor in anchorList:
            dist = torch.dist(out, anchor)
            print(data[1], dist)
        # correctNum += 1
    # print("Test correct rate: %.3f" % (correctNum / len(datas)))

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir = "./model"
    try:
        fileName = sorted(os.listdir(dir), reverse=True)[0]
        model_path = dir + '/' + fileName
        print("加载模型：", model_path)
        model = torch.load(model_path).to(device)
        print("开始测试数据。。。")
        test_loader, anchors = getFaceDatas.getTestDataSet(1)
        test(model, test_loader, anchors, device)
    except:
        print("模型不存在，先在train.py训练模型")