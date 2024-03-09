# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/8 17:26
# Description:
import matplotlib.pyplot as plt
import torch.cuda
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torchvision import datasets
from imgaug import augmenters as iaa  # 用于图像增强
import cv2

data_folder = "../data/FMNIST"

device = "cuda" if torch.cuda.is_available() else "cpu"


class FMNISTDataset(Dataset):
    def __init__(self, x, y, aug=None):
        # 每个图片扁平化，并将其形状改变为[batch_size, 784]
        # 像素归一化
        x = x.float() / 255.
        self.x = x.view(-1, 1, 28 * 28)
        self.y = y
        self.aug = aug

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)

    # 转化并进行数据增强
    def collate_fn(self, batch):
        ims, classes = list(zip(*batch))
        if self.aug: ims = self.aug.augment_image(images=ims)
        # (num_images, height, width) => (num_images, 1, height, width)
        ims = torch.tensor(ims)[:, None, :, :].to(device) / 255
        classes = torch.tensor(classes).to(device)
        return ims, classes


# 构建图像的训练模型
def get_model():
    model = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),  # 1* 28 * 28 -> 64 * 26 * 26
                          nn.MaxPool2d(2),  # 64 * 26 * 26 -> 64 * 13 * 13
                          nn.ReLU(),
                          nn.Conv2d(64, 128, kernel_size=3),  # 64 * 13 * 13 -> 128 * 11 * 11
                          nn.MaxPool2d(2),  # 128 * 11 * 11 -> 128 * 5 * 5 = 3200
                          nn.ReLU(),
                          nn.Flatten(),
                          nn.Linear(3200, 256),
                          nn.ReLU(),
                          nn.Linear(256, 10)).to(device)
    # 定义交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()
    # 优化器
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer


def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


# 数据增强
aug = iaa.Sequential([iaa.Affine(translate_px={'x': {(-10, 10)}}, mode='constant')])


def get_data(batch_size=64):
    train_fashion_mnist = datasets.FashionMNIST(data_folder, download=True, train=True)
    valid_fashion_mnist = datasets.FashionMNIST(data_folder, download=True, train=False)

    train = FMNISTDataset(train_fashion_mnist.data, train_fashion_mnist.targets, aug=aug)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=train.collate_fn)

    valid = FMNISTDataset(valid_fashion_mnist.data, valid_fashion_mnist.targets, aug=aug)
    # 直接一次验证数据集
    valid_dl = DataLoader(valid, batch_size=len(valid_fashion_mnist.data), shuffle=True, collate_fn=valid.collate_fn)

    return train_dl, valid_dl


# 图片增强的方式
# 当然我们可以使用seq = iaa.Sequential(iaa.Affine(scale=2, fit_output=True),
#                             iaa.Dropout(p=0.2))
#   seq.augment_image(image) 进行多操作图片
def image_strong_style(image):
    fig, ax = plt.subplots(3, 6, figsize=(12, 5))
    ax = ax.flatten()
    ax[0].imshow(iaa.Affine(scale=2, fit_output=True).augment_image(image), cmap='gray')
    ax[1].imshow(iaa.Affine(translate_px=10).augment_image(image), cmap='gray')  # 一定像素平移  图片上下都移动10px
    ax[2].imshow(iaa.Affine(translate_px={'x': 10, 'y': 2}).augment_image(image), cmap='gray')  # 自定义上下偏移多少距离
    ax[3].imshow(iaa.Affine(rotate=30, fit_output=True).augment_image(image), cmap='gray')  # 旋转30度 默认填充黑色
    ax[4].imshow(iaa.Affine(rotate=30, fit_output=True, cval=255).augment_image(image),
                 cmap='gray')  # 旋转30度 不属于该图像的部分填充白色
    ax[5].imshow(iaa.Affine(shear=30, fit_output=True).augment_image(image), cmap='gray')  # 对图像的某个部分进行旋转

    ax[6].imshow(iaa.Affine(rotate=45, fit_output=True, mode='constant').augment_image(image),
                 cmap='gray')  # 填充某个常数值
    ax[7].imshow(iaa.Affine(rotate=45, fit_output=True, mode='edge').augment_image(image),
                 cmap='gray')  # 填充边缘值
    ax[8].imshow(iaa.Affine(rotate=45, fit_output=True, mode='symmetric').augment_image(image),
                 cmap='gray')  # 沿数组边缘镜像反射
    ax[9].imshow(iaa.Affine(rotate=45, fit_output=True, mode='reflect').augment_image(image),
                 cmap='gray')  # 沿每个轴的第一和最后一个值进行镜像反射
    ax[10].imshow(iaa.Affine(rotate=45, fit_output=True, mode='wrap').augment_image(image),
                  cmap='gray')  # 沿着轴缠绕

    ax[11].imshow(iaa.Multiply(0.5).augment_image(image), cmap='gray')  # 每个像素值都乘于0.5 变暗了
    # 127 + a * (像素-127) 高像素值减少，低像素值增加 变亮了
    ax[12].imshow(iaa.LinearContrast(0.5).augment_image(image), cmap='gray')
    # 高斯模糊 随着sigma的增加越来越模糊
    ax[13].imshow(iaa.GaussianBlur(sigma=4).augment_image(image), cmap='gray')

    # 添加噪声
    ax[14].imshow(iaa.Dropout(p=0.2).augment_image(image), cmap='gray')  # 随机丢弃20%的像素 即填充黑色
    ax[15].imshow(iaa.SaltAndPepper(0.2).augment_image(image), cmap='gray')  # 随机20%像素填充黑色或白色

    plt.show()


if __name__ == '__main__':
    image = cv2.imread("../images/Hemanvi.jpeg")
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_strong_style(image)
