#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/22 19:07
# @Author : LYX-夜光

import numpy as np

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 激活函数求导
def dSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 均方误差
def loss_MSE(yTrue, yPred):
    return ((yTrue - yPred) ** 2).mean()

# 神经网络
class NeuralNetworks():
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, yTrues):
        learnRate = 0.5
        epochs = 2000
        for epoch in range(epochs):  # 每一次迭代
            for x, yTrue in zip(data, yTrues):  # 每一个输入数据
                # 计算每一层节点
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1  # 输入层至隐藏层的线性变换
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h1, h2 = sigmoid(sum_h1), sigmoid(sum_h2)  # 经过激活函数
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3  # 隐藏层至输出的线性变换
                o1 = sigmoid(sum_o1)  # 经过激活函数
                yPred = o1  # 输出数据

                # 损失函数的求导
                dL_dyPred = -2 * (yTrue - yPred)

                # 输出层对隐藏层做链式法则求偏导
                dyPred_dw5 = h1 * dSigmoid(sum_o1)
                dyPred_dw6 = h2 * dSigmoid(sum_o1)
                dyPred_db3 = dSigmoid(sum_o1)

                dyPred_dh1 = self.w5 * dSigmoid(sum_o1)
                dyPred_dh2 = self.w6 * dSigmoid(sum_o1)

                # 隐藏层对输入层做链式法则求偏导
                # 对神经元h1
                dh1_dw1 = x[0] * dSigmoid(sum_h1)
                dh1_dw2 = x[1] * dSigmoid(sum_h1)
                dh1_db1 = dSigmoid(sum_h1)

                # 对神经元h2
                dh2_dw3 = x[0] * dSigmoid(sum_h2)
                dh2_dw4 = x[1] * dSigmoid(sum_h2)
                dh2_db2 = dSigmoid(sum_h2)

                # 神经元o1，梯度下降
                self.w5 -= learnRate * dL_dyPred * dyPred_dw5
                self.w6 -= learnRate * dL_dyPred * dyPred_dw6
                self.b3 -= learnRate * dL_dyPred * dyPred_db3

                # 神经元h1，梯度下降
                self.w1 -= learnRate * dL_dyPred * dyPred_dh1 * dh1_dw1
                self.w2 -= learnRate * dL_dyPred * dyPred_dh1 * dh1_dw2
                self.b1 -= learnRate * dL_dyPred * dyPred_dh1 * dh1_db1

                # 神经元h2，梯度下降
                self.w3 -= learnRate * dL_dyPred * dyPred_dh2 * dh2_dw3
                self.w4 -= learnRate * dL_dyPred * dyPred_dh2 * dh2_dw4
                self.b2 -= learnRate * dL_dyPred * dyPred_dh2 * dh2_db2

            if epoch % 100 == 0:
                yPreds = np.apply_along_axis(self.feedforward, 1, data)  # 每行数据计算预测值
                loss = loss_MSE(yTrues, yPreds)  # 计算损失函数值
                print("Epoch %d loss: %.3f" % (epoch, loss))


if __name__ == "__main__":
    data = np.array([[133, 65],  # 每一行是体重（磅）和身高（英寸）
                     [160, 72],
                     [152, 70],
                     [120, 60]])
    maxD, minD = data.max(0), data.min(0)
    newData = (data - minD) / (maxD - minD)  # 单位化

    yTrues = np.array([0, 1, 1, 0])  # 1男性 0女性
    network = NeuralNetworks()
    network.train(newData, yTrues)

    # 测试样例
    Alice = np.array([128, 63])
    gender = network.feedforward((Alice - minD) / (maxD - minD))
    print("Alice: %f" % gender)

    Frank = np.array([155, 68])
    gender = network.feedforward((Frank - minD) / (maxD - minD))
    print("Frank: %f" % gender)