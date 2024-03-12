# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/11 20:47
# Description: 多任务学习: 就是一个模型可以预测多个属性，例: 根据图片预测 年龄和性别

import torch
import numpy as np, cv2, pandas as pd, glob, time
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.vgg import VGG16_Weights
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# https://github.com/joojs/fairface?tab=readme-ov-file
# Data 部分下载数据集

# 数据集读取
train_df = pd.read_csv("../data/fairface-label-train.csv")
val_df = pd.read_csv("../data/fairface-label-val.csv")

IMAGE_SIZE = 224
age_map = {"0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
           "40-49": 5, "50-59": 6, "60-69": 7, "70+": 8}
gender_map = {"Female": 0, "Male": 1}


class GenderAgeClass(Dataset):
    def __init__(self, df, img_path="../images/"):
        self.df = df
        self.img_path = img_path
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # squeeze()函数被用来减少选取出来的数据结构的维度, 确保提取出来的数据是一个标量值或者一维的Series
        f = self.df.iloc[item].squeeze()
        # 文件的位置信息
        file = f.file
        gen = gender_map[f.gender]
        # age 分为 0-2 3-9 10-19 20-29 ... 70+
        # 转换为数字
        age = age_map[f.age]
        im = cv2.imread(self.img_path + file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, age, gen

    def preprocess_image(self, im):
        im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
        im = torch.tensor(im).permute(2, 0, 1)
        im = self.normalize(im / 255.)
        # 如果im是一个形状为(height, width, channels)的三维数组（例如，一个RGB图像），
        #       那么im[None]将创建一个新的维度，使其形状变为(1, height, width, channels)
        # 这通常用于将单个图像转换为一个批次中的单个图像，以便与期望接收批次数据的模型或函数兼容。
        return im[None]

    # 加工数据处理为tensor
    def collate_fn(self, batch):
        images, ages, genders = [], [], []
        for im, age, gender in batch:
            im = self.preprocess_image(im)
            images.append(im)
            ages.append(float(age))
            genders.append(float(gender))
        ages, genders = [torch.tensor(x).to(device).float() for x in [ages, genders]]
        images = torch.cat(images).to(device)
        return images, ages, genders


def get_data():
    train_age_gender = GenderAgeClass(train_df)
    valid_age_gender = GenderAgeClass(val_df)

    train_dataloader = DataLoader(train_age_gender, batch_size=32, shuffle=True, drop_last=True,
                                  collate_fn=train_age_gender.collate_fn)
    valid_dataloader = DataLoader(valid_age_gender, batch_size=32, shuffle=True, drop_last=True,
                                  collate_fn=valid_age_gender.collate_fn)
    return train_dataloader, valid_dataloader, train_age_gender


class AgeGenderClassifier(nn.Module):
    def __init__(self):
        super(AgeGenderClassifier, self).__init__()
        self.intermediate = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.age_classifier = nn.Linear(64, 9)
        self.gender_classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.intermediate(x)
        age = torch.argmax(self.age_classifier(x), dim=1)
        gender = torch.argmax(self.gender_classifier(x), dim=1)
        return gender, age


def get_model():
    model = models.vgg16(weights=VGG16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3), nn.MaxPool2d(2), nn.ReLU(), nn.Flatten())
    # -1, 512, 5,5 -> -1 512, 2, 2

    model.classifier = AgeGenderClassifier()

    # 分类模型 对于损失函数的计算也分别拆开
    gender_criterion = nn.BCELoss()  # 二元交叉熵
    age_criterion = nn.CrossEntropyLoss()  # 如果预测年龄值 使用回归损失 nn.MSELoss() 预测年龄范围使用cross_entry_loss
    loss_func = gender_criterion, age_criterion

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return model.to(device), loss_func, optimizer


model, criterion, optimizer = get_model()


def train_batch(data, model, optimizer, criteria):
    model.train()
    images, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(images)
    gender_criterion, age_criterion = criteria
    # 性别损失
    gender_loss = gender_criterion(pred_gender, gender)
    # 年龄损失
    age_loss = age_criterion(pred_age, age)
    total_loss = gender_loss + age_loss
    total_loss.backward()
    optimizer.step()
    return total_loss


def validate_batch(data, model, criteria):
    model.eval()
    images, age, gender = data
    with torch.no_grad():
        pred_gender, pred_age = model(images)

    gender_criterion, age_criterion = criteria
    # 性别损失
    gender_loss = gender_criterion(pred_gender, gender)
    # 年龄损失
    age_loss = age_criterion(pred_age, age)
    total_loss = gender_loss + age_loss
    # 计算性别
    _, gender_pred = torch.max(pred_gender, 1)
    gender_acc = (gender_pred == gender).float().mean().item()

    # 计算年龄范围
    _, age_pred = torch.max(pred_age, 1)
    age_acc = (age_pred == age).float().mean().item()
    return total_loss.item(), gender_acc, age_acc


def train(train_loader, test_loader):
    model, criteria, optimizer = get_model()
    val_gender_accuracies, val_age_accuracies, train_losses, val_losses = [], [], [], []
    n_epochs = 5
    start = time.time()
    for epoch in range(n_epochs):
        epoch_train_loss, epoch_test_loss = 0, 0
        val_age_acc, val_gender_acc, nums = 0, 0, 0
        _n = len(train_loader)

        for ix, data in enumerate(train_loader):
            loss = train_batch(data, model, optimizer, criteria)
            epoch_train_loss += loss.item()

        for ix, data in enumerate(test_loader):
            loss, gender_acc, age_acc = validate_batch(data, model, criteria)
            epoch_test_loss += loss
            val_age_acc += age_acc
            val_gender_acc += gender_acc
            nums += len(data[0])

        val_age_acc /= nums
        val_gender_acc /= nums
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)

        elapsed = time.time() - start
        print("{}/{} ({:.2f}s - {:.2f}s) remaining".format(epoch + 1, n_epochs, time.time() - start,
                                                           (n_epochs - epoch) * (elapsed / (epoch + 1))))

        info = f'''Epoch: {epoch + 1:03d}\tTrain Loss: {epoch_train_loss:.3f}\tTest: {epoch_test_loss:.3f}'''
        info += f'\nGender Accuracy: {val_gender_acc * 100:.2f}%\tAge acc: {val_age_acc:.2f}\n'
        print(info)

        val_gender_accuracies.append(val_gender_acc)
        val_age_accuracies.append(val_age_acc)
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_test_loss)
    return train_losses, val_losses, val_gender_accuracies, val_age_accuracies


def plot_image(n_epochs, val_gender_accuracies, val_age_accuracies):
    epochs = np.arange(1, (n_epochs + 1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax = ax.flatten()
    ax[0].plot(epochs, val_gender_accuracies, 'bo')
    ax[1].plot(epochs, val_age_accuracies, 'r')

    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('MAE')
    ax[0].set_title('Validation Gender Accuracy')
    ax[1].set_title('Validation Age Mean-Absolute-Error')
    plt.show()


def find_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


# 画出验证的图片
def plot_valid_image(train_age_gender):
    # 图片需要下载
    im = cv2.imread('../images/5_9.JPG')
    im = train_age_gender.preprocess_image(im).to(device)
    gender, age = model(im)
    pred_gender = gender.to('cpu').detach().numpy()
    pred_age = age.to('cpu').detach().numpy()
    im = cv2.imread('../images/5_9.JPG')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    print('predicted gender:', find_key_by_value(gender_map, torch.argmax(pred_gender[0])), ' Predicted age range:',
          find_key_by_value(age_map, torch.argmax(pred_age[0])))


train_dataloader, valid_dataloader, train_age_gender = get_data()
model, loss_func, optimizer = get_model()
train_losses, val_losses, val_gender_accuracies, val_age_accuracies = train(train_dataloader, valid_dataloader)
plot_image(5, val_gender_accuracies, val_age_accuracies)

plot_valid_image(train_age_gender)
