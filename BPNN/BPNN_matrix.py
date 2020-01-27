#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/24 16:34
# @Author : LYX-夜光

import numpy as np

# 逻辑回归函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 逻辑回归函数的导数
def dSigmoid(x):
    return np.multiply(sigmoid(x), (1 - sigmoid(x)))

# 均方误差
def loss_MSE(yTrue, yPred):
    return ((yTrue - yPred).A**2).mean()

# 神经网络
class NeuralNetworks():
    def __init__(self, xMat, yMat, hidden_layer=2):
        self.x, self.y = np.mat(xMat), np.mat(yMat)
        input_data = self.x.shape[1]  # 输入层个数
        output_data = self.y.shape[1]  # 输出层个数

        # 输入层->隐藏层，隐藏层->输出层的权值
        self.w1 = np.mat(np.random.randn(input_data + 1, hidden_layer))
        self.w2 = np.mat(np.random.randn(hidden_layer + 1, output_data))

    def feedforward(self, x, actiFunc=sigmoid):
        x = np.c_[np.ones(x.shape[0]), x]
        h = actiFunc(x * self.w1)  # 输入层至隐藏层的线性变换
        h = np.c_[np.ones(h.shape[0]), h]
        o = actiFunc(h * self.w2)  # 隐藏层至输出层的线性变换
        return o

    def train(self, actiFunc=sigmoid, dActiFunc=dSigmoid):
        learnRate = 0.5
        epochs = 2000
        for epoch in range(epochs):  # 每一次迭代
            for i in range(self.x.shape[0]):  # 每一个输入数据
                # 计算每一层节点
                x = np.c_[[1], self.x[i, :]]
                sum_h = x * self.w1  # 输入层至隐藏层的线性变换
                h = actiFunc(sum_h)  # 计算隐藏层的节点
                h = np.c_[[1], h]
                sum_o = h * self.w2  # 隐藏层至输出层的线性变换
                o = actiFunc(sum_o)  # 计算输出层的节点

                # 损失函数的求导
                dL_do = -2 * (self.y[i, :] - o)

                # 输出层节点对隐藏层做链式法则求偏导
                do_dw2 = h.T * dActiFunc(sum_o)
                do_dh = np.mat(self.w2[1:].A * dActiFunc(sum_o).A).T

                # 隐藏层节点对输入层做链式法则求偏导
                dh_dw1 = x.T * dActiFunc(sum_h)

                # 梯度下降
                self.w2 -= np.mat(learnRate * dL_do.A * do_dw2.A)
                self.w1 -= np.mat(learnRate * dL_do.A * do_dh.A * dh_dw1.A)

            if epoch % 100 == 0:
                yPreds = self.feedforward(self.x)  # 每行数据计算预测值
                loss = loss_MSE(self.y, yPreds)  # 计算损失函数值
                print("Epoch %d loss: %.3f" % (epoch, loss))

if __name__ == "__main__":
    data = np.array([[133, 65],  # 每一行是体重（磅）和身高（英寸）
                     [160, 72],
                     [152, 70],
                     [120, 60]])
    dataMean, dataStd = data.mean(), data.std()
    newData = (data - dataMean) / dataStd  # 标准化
    yTrues = np.array([[0, 1, 1, 0]]).T  # 1男性 0女性

    network = NeuralNetworks(newData, yTrues)
    network.train()

    # 测试样例
    Alice = np.array([128, 63])
    gender = network.feedforward(np.mat((Alice - dataMean) / dataStd))
    print("Alice: %f" % gender)

    Frank = np.array([155, 68])
    gender = network.feedforward(np.mat((Frank - dataMean) / dataStd))
    print("Frank: %f" % gender)