# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/11 10:29
# Description: 迁移学习使用VGG16
import torch.cuda
import torch.nn as nn
from torchvision import transforms, models
from torchsummary import summary
from torchvision.models.vgg import VGG16_Weights
from common import CatDogPred

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model():
    # 先导入vgg16的模型，再修改最后几层的参数信息
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False
    # [-1, 512, 14, 14]         2,359,808 -> 原来 [-1, 512, 7, 7]  (7,7)
    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1),
                                     nn.Sigmoid())
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 查看模型的基本信息
    summary(model, torch.zeros(1, 3, 224, 224))

    return model.to(device), loss_fn, optimizer


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_dl, valid_dl = CatDogPred.get_data(normalize)
model, loss_fn, optimizer = get_model()
train_losses, train_accuracies, val_accuracies = CatDogPred.train_get_loss_accuracies(model, train_dl, valid_dl)
CatDogPred.plot(train_accuracies, val_accuracies)
