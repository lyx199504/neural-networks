#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/1/24 11:47
# @Author : LYX-夜光

import numpy as np
import sys

from skimage import data, color, io

# 卷积层
def conv(img, convFilter):
    if len(img.shape) != len(convFilter.shape) - 1:
        print("图片和过滤器维度不匹配！")
        sys.exit()
    if len(img.shape) > 2 or len(convFilter.shape) > 3:
        if img.shape[-1] != convFilter.shape[-1]:
            print("彩色图片和过滤器的通道数不一致！")
            sys.exit()
    if convFilter.shape[1] != convFilter.shape[2]:
        print("过滤器必须是方阵！")
        sys.exit()
    if convFilter.shape[1] % 2 == 0:
        print("过滤器方阵必须是奇数方阵！")
        sys.exit()
    # 卷积结果
    featureMaps = np.zeros((img.shape[0] - convFilter.shape[1] + 1,
                            img.shape[1] - convFilter.shape[2] + 1,
                            convFilter.shape[0]))
    # 计算卷积外循环
    for filterNum in range(convFilter.shape[0]):
        curFilter = convFilter[filterNum]
        if len(curFilter.shape) > 2:  # 彩色图片的过滤器
            convMap = cal_conv(img[:, :, 0], curFilter[:, :, 0])
            for i in range(1, curFilter.shape[-1]):
                convMap += cal_conv(img[:, :, i], curFilter[:, :, i])
        else:  # 灰度图片过滤器
            convMap = cal_conv(img, curFilter)
        featureMaps[:, :, filterNum] = convMap
    return featureMaps

# 计算卷积层
def cal_conv(img, curFilter):
    filterSize = curFilter.shape[0]  # 过滤器大小
    convResult = np.zeros((img.shape[0] - curFilter.shape[0] + 1, img.shape[1] - curFilter.shape[1] + 1))
    for r in range(convResult.shape[0]):
        for c in range(convResult.shape[1]):
            # 过滤每个与过滤器大小一致的子图
            subImg = img[r: r+filterSize, c: c+filterSize]
            convResult[r, c] = np.sum(subImg * curFilter)
    return convResult

# 线性整流函数
def relu(featureMaps):
    reluResults = np.zeros(featureMaps.shape)
    for mapNum in range(featureMaps.shape[-1]):
        for r in range(featureMaps.shape[0]):
            for c in range(featureMaps.shape[1]):
                reluResults[r, c, mapNum] = np.max(featureMaps[r, c, mapNum], 0)
    return reluResults

# 池化层
def pooling(featureMaps, size=2, stride=2):
    poolResults = np.zeros((np.int(np.ceil((featureMaps.shape[0] - size)/stride + 1)),
                            np.int(np.ceil((featureMaps.shape[1] - size)/stride + 1)),
                            featureMaps.shape[-1]))
    for mapNum in range(featureMaps.shape[-1]):
        for r in range(poolResults.shape[0]):
            for c in range(poolResults.shape[1]):
                rf, cf = r*stride, c*stride  # 将矩阵压缩
                poolResults[r, c, mapNum] = np.max(featureMaps[rf: rf+size, cf: cf+size, mapNum])
    return poolResults

if __name__ == "__main__":
    img = data.chelsea()
    img = color.rgb2gray(img)

    layerFilter1 = np.zeros((2, 3, 3))  # 2个初始过滤器
    layerFilter1[0, :, 0] = layerFilter1[1, 2, :] = -1
    layerFilter1[0, :, 2] = layerFilter1[1, 0, :] = 1
    featureMaps1 = conv(img, layerFilter1)  # 卷积计算
    reluResults1 = relu(featureMaps1)  # 对卷积结果使用激活函数
    poolResults1 = pooling(reluResults1)  # 对激活结果进行池化

    layerFilter2 = np.random.rand(3, 5, 5, poolResults1.shape[-1])
    featureMaps2 = conv(poolResults1, layerFilter2)
    reluResults2 = relu(featureMaps2)
    poolResults2 = pooling(reluResults2)

    layerFilter3 = np.random.rand(1, 7, 7, poolResults2.shape[-1])
    featureMaps3 = conv(poolResults2, layerFilter3)
    reluResults3 = relu(featureMaps3)
    poolResults3 = pooling(reluResults3)

    img = poolResults1[:, :, 0]
    io.imshow(img)
    io.show()