#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/21 17:15
# @Author : LYX-夜光

import torch
import torchvision

# facenet模型
class FaceNet(torch.nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512 * 4 * 4, 128)  # 新增一层全连接

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)

        alpha = 10
        self.features = self.features * alpha
        return self.features