# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/8 20:30
# Description: 猫狗预测
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
from torch import optim
import cv2, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import DataLoader, Dataset
from random import shuffle, seed
from torchsummary import summary
import matplotlib.ticker as mticker

device = "cuda" if torch.cuda.is_available() else "cpu"

seed(10)

train_data_dir = "../data/training_set/training_set"
test_data_dir = "../data/test_set/test_set"


class cats_dogs(Dataset):
    def __init__(self, folder):
        # glob函数会返回一个列表，这个列表包含了所有匹配到的文件路径
        cats = glob(folder + '/cats/*.jpg')
        dogs = glob(folder + '/dogs/*.jpg')
        self.paths = cats + dogs
        shuffle(self.paths)
        # 以dog开头的记为1 否则为0
        self.targets = [fpath.split("/")[-1].startswith("dog") for fpath in self.paths]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        f = self.paths[item]
        target = self.targets[item]
        # cv2读取出来的格式为[height, width, 通道]  通道为BGR   ::-1就是将通道换为RGB
        im = (cv2.imread(f)[:, :, ::-1])
        im = cv2.resize(im, (224, 224))
        # permute(2,0,1) 就是把通道提到前面
        return torch.tensor(im / 255).permute(2, 0, 1).to(device=device).float(), \
            torch.tensor([target]).float().to(device)


def conv_layer(ni, no, kernel_size, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(2)
    )


def get_model():
    model = nn.Sequential(
        conv_layer(3, 64, 3), conv_layer(64, 512, 3), conv_layer(512, 512, 3),
        conv_layer(512, 512, 3), conv_layer(512, 512, 3), conv_layer(512, 512, 3),
        nn.Flatten(), nn.Linear(512, 1), nn.Sigmoid()).to(device)
    # 因为二分类问题所以最后输出1
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def get_data():
    train = cats_dogs(train_data_dir)
    train_dl = DataLoader(train, batch_size=32, shuffle=True, drop_last=True)

    valid = cats_dogs(test_data_dir)
    valid_dl = DataLoader(valid, batch_size=32, shuffle=True, drop_last=True)
    return train_dl, valid_dl


def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


@torch.no_grad()
def accuracy(x, y, model):
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()


@torch.no_grad()
def val_loss(x, y, model):
    prediction = model(x)
    val_loss = loss_fn(prediction, y)
    return val_loss.item()


def train_get_loss_accuracies(model, train_dataloader, valid_dataloader):
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    for epoch in range(5):
        train_epoch_losses, train_epoch_accuracies = [], []
        val_epoch_accuracies = []

        for ix, batch in enumerate(iter(train_dataloader)):
            x, y = batch
            batch_loss = train_batch(x, y, model, optimizer, loss_fn)
            train_epoch_losses.append(batch_loss)

        train_epoch_loss = np.mean(train_epoch_losses)

        for ix, batch in enumerate(iter(train_dataloader)):
            x, y = batch
            is_correct = accuracy(x, y, model)
            train_epoch_accuracies.extend(is_correct)

        train_epoch_accuracy = np.mean(train_epoch_accuracies)

        # 计算验证集准确率
        for ix, batch in enumerate(iter(valid_dataloader)):
            x, y = batch
            valid_is_correct = accuracy(x, y, model)
            # valid_is_correct追加到val_epoch_accuracies尾部
            val_epoch_accuracies.extend(valid_is_correct)

        val_epoch_accuracy = np.mean(val_epoch_accuracies)

        # 记录训练集验证集的误差和准确率
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        val_accuracies.append(val_epoch_accuracy)
    return train_losses, train_accuracies, val_accuracies


def plot(train_accuracies, val_accuracies):
    epochs = np.arange(5) + 1
    plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Valid Accuracy')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title("Training and validation accuracy \n with 4k data for train")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.gca().set_yticklabels(["{:.0f}%".format(x * 100) for x in plt.gca().get_yticks()])
    plt.legend()
    plt.grid("off")
    plt.show()


model, loss_fn, optimizer = get_model()
summary(model, torch.zeros(1, 3, 224, 224))
train_dl, valid_dl = get_data()
train_losses, train_accuracies, val_accuracies = train_get_loss_accuracies(model, train_dl, valid_dl)
plot(train_accuracies, val_accuracies)
