#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/2/10 0:40
# @Author : LYX-夜光

import os
import torch
from CNN.faceRecognition import getFaceDatas, faceNet

# 测试
def test(model, datas, anchor, device):
    model.eval()  # 训练模式关闭，使model不再改变权值
    image = torch.tensor([anchor[0].tolist()]).to(device)
    with torch.no_grad():
        anchor = model(image).cpu()
    TP, TN, FP, FN = 0.0, 0.0, 0.0, 0.0
    divide = 10.0  # 观察结果后选择的一个划分值
    for data in datas:
        image = data[0].to(device)
        with torch.no_grad():
            out = model(image).cpu()
        dist = torch.dist(out, anchor)
        # print(data[1], dist)
        if data[1] == 0 and dist < divide:
            TP += 1
        if data[1] != 0 and dist >= divide:
            TN += 1
        if data[1] != 0 and dist < divide:
            FP += 1
        if data[1] == 0 and dist >= divide:
            FN += 1
    print("测试结果如下。。。")
    print("正确率: %.6f" % ((TP + TN) / len(datas)))
    print("查准率：%.6f" % (TP / (TP + FP)))
    print("查全率：%.6f" % (TP / (TP + FN)))

if __name__ == "__main__":
    '''
    测试集，结果还行，准确率和查准率高，但是查全率较低，比较严格
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dir = "./model"
    try:
        fileName = sorted(os.listdir(dir), reverse=True)[0]
        model_path = dir + '/' + fileName
        print("加载模型：", model_path)
        model = torch.load(model_path).to(device)
    except:
        print("模型不存在，使用初始模型：facenet")
        model = faceNet.FaceNet().to(device)
    print("开始测试数据。。。")
    test_loader, anchors = getFaceDatas.getTestDataSet(1)
    test(model, test_loader, anchors, device)