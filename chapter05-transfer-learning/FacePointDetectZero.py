# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/11 12:27
# Description: 人脸关键点检测

import torchvision
import torch.nn as nn
import torch
from torchvision.models.vgg import VGG16_Weights
import matplotlib.ticker as mticker
from torchvision import transforms, models, datasets
from torchsummary import summary
import numpy as np, pandas as pd, os, glob, cv2
from torch.utils.data import TensorDataset, DataLoader, Dataset
from copy import deepcopy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

root_dir = "../data/P1_Facial_Keypoints/data/training/"
all_img_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
# 第0列是图片的名字，1: 之后是该图片的关键点信息
data = pd.read_csv("../data/P1_Facial_Keypoints/data/training_frames_keypoints.csv")


class FaceData(Dataset):
    def __init__(self, df):
        super(FaceData).__init__()
        self.df = df
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2, 0, 1)
        img = self.normalize(img).float()
        return img.to(device)

    def __getitem__(self, item):
        img_path = "../data/P1_Facial_Keypoints/data/training/" + self.df.iloc[item, 0]
        img = cv2.imread(img_path) / 255
        # 将期望输出值归一化为适合于原始图像大小的比例 1: 之后是该图片的关键点数据 每个图片的关键点数据都是136个
        kp = deepcopy(self.df.iloc[item, 1:].tolist())
        print(len(kp))
        # kp[0::2] 使用切片操作来提取 kp 列表中所有偶数索引的元素，即所有x坐标
        # img.shape[1] 通常是图像的宽度，所以这一步是将x坐标从像素值转换为0到1之间的值
        kp_x = (np.array(kp[0::2]) / img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2]) / img.shape[0]).tolist()
        img = self.preprocess_input(img)
        kp2 = torch.tensor(kp_x + kp_y)
        return img, kp2

    def load_img(self, ix):
        img_path = "../data/P1_Facial_Keypoints/data/training/" + self.df.iloc[ix, 0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (224, 224))
        return img


def get_data():
    train, test = train_test_split(data, test_size=0.2, random_state=101)
    # 删除原来的索引列 并重新使用[0,1,2,...]作为索引下标
    train_dataset = FaceData(train.reset_index(drop=True))
    test_dataset = FaceData(test.reset_index(drop=True))

    train_dataloader = DataLoader(train_dataset, batch_size=32)
    test_dataloader = DataLoader(test_dataset, batch_size=32)

    return train_dataloader, test_dataloader, test_dataset


def get_model():
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    # [-1, 512, 14, 14]         2,359,808 -> 原来 [-1, 512, 7, 7]  (7,7)
    model.avgpool = nn.Sequential(nn.Conv2d(512, 512, 3),
                                  nn.MaxPool2d(2),
                                  nn.Flatten())

    model.classifier = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, 136),
                                     nn.Sigmoid())
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 查看模型的基本信息
    summary(model, torch.zeros(1, 3, 224, 224))

    return model.to(device), criterion, optimizer


def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss


def validate_batch(img, kps, model, criterion):
    model.eval()
    with torch.no_grad():
        _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss


def train_model_loss(train_loader, test_loader, model, optimizer, criterion):
    train_loss, test_loss = [], []
    n_epochs = 50

    for epoch in range(n_epochs):
        print(f" epoch {epoch + 1} : 50")
        epoch_train_loss, epoch_test_loss = 0, 0
        for ix, (img, kps) in enumerate(train_loader):
            loss = train_batch(img, kps, model, optimizer=optimizer, criterion=criterion)
            epoch_train_loss += loss.item()
        epoch_train_loss /= (ix + 1)

        for ix, (img, kps) in enumerate(test_loader):
            ps, loss = validate_batch(img, kps, model, criterion=criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= (ix + 1)

        train_loss.append(epoch_train_loss)
        test_loss.append(epoch_test_loss)
    return train_loss, test_loss


def plot(train_loss, val_loss, info):
    epochs = np.arange(5) + 1
    plt.plot(epochs, train_loss, 'bo', label='Training Accuracy')
    plt.plot(epochs, val_loss, 'r', label='Valid Accuracy')
    plt.title("Training and validation loss \n with " + info + " for train")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid("off")
    plt.show()


def plot_key_point_image(test_dataset, ix, model):
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title("Origin image")
    im = test_dataset.load_img(ix)
    plt.imshow(im)
    plt.grid(False)

    plt.subplot(222)
    plt.title("Image with facial keypoints")
    x, _ = test_dataset[ix]
    plt.imshow(im)
    kp = model(x[None]).flatten().detach().cpu()
    plt.scatter(kp[:68] * 224, kp[68:] * 224, c='r')
    plt.grid(False)
    plt.show()


train_dataloader, test_loader, test_dataset = get_data()
model, criterion, optimizer = get_model()
train_loss, test_loss = train_model_loss(train_dataloader, test_loader, model, optimizer, criterion)
plot(train_loss, test_loss, "vgg16")
# 画出一个张图片的关键点信息
plot_key_point_image(test_dataset, 0, model)
