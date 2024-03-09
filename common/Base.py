# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/7 15:54
# Description:
from copy import deepcopy
import numpy as np


# 设置前向传播的函数
def feed_forward(inputs, outputs, weights):
    # 计算预测值中间值
    pre_hidden = np.dot(inputs, weights[0]) + weights[1]
    # 激活值
    activation = 1 / (1 + np.exp(-pre_hidden))
    # 预测值
    pre_out = np.dot(activation, weights[2]) + weights[3]
    # 计算误差损失值
    mean_squared_error = np.mean(np.square(pre_out - outputs))
    return mean_squared_error


def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)

    # 计算原始的损失值
    original_loss = feed_forward(inputs, outputs, original_weights)
    for i, layer in enumerate(original_weights):
        # 遍历每层中的参数信息
        for index, weight in np.ndenumerate(layer):
            temp_weights[i][index] += 0.0001
            _loss_plus = feed_forward(inputs, outputs, temp_weights)
            grad = (_loss_plus - original_loss) / 0.0001
            updated_weights[i][index] -= grad * lr
    return updated_weights, original_loss


x = np.array([1, 1])
y = np.array([[0]])
W = [
    np.array([[-0.0053, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896]], dtype=np.float32).T,
    np.array([-0.0140, 0.5607, -0.0628], dtype=np.float32),
    np.array([[0.1528, -0.1745, -0.1135]], dtype=np.float32).T,
    np.array([-0.5516], dtype=np.float32)
]

losses = []

for epoch in range(100):
    W, loss = update_weights(x, y, weights=W, lr=0.01)
    losses.append(loss)

print(W[0].shape)