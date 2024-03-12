# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/12 15:38
# Description:
# 使用训练好的CNN模型对输入图像进行前向传播，并提取最后一个卷积层的特征图。
# 计算类别特定的特征图权重，通常是通过全局平均池化（Global Average Pooling）的方式得到。
# 将特征图与权重相乘并进行加权求和，得到CAM。
# 将CAM进行上采样（upsampling）至原始输入图像的大小，并进行归一化处理。
# 将CAM叠加在原始输入图像上，以可视化CNN模型对不同区域的关注程度。
import torch
# 数据集网站
# https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria/code
from torch_snippets import *
from torchvision import transforms
import torch
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

# 指定与输出类别对应的索引
id2int = {"Parasitized": 0, "Uninfected": 1}

# 训练处理
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ColorJitter(brightness=(0.95, 1.05), contrast=(0.95, 1.05), saturation=(0.95, 1.05),
                           hue=0.05),
    transforms.RandomAffine(5, translate=(0.01, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 验证集的图像处理
valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# 定义数据集
class MalariaImages(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
        logger.info(len(self))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        fpath = self.files[item]
        # 读取图片父级目录的信息 cell_images/Parasitized/001.jpg 读取的就是Parasitized
        clss = fname(parent(fpath))
        # mode = 1 表示将bgr 换为rgb
        img = read(fpath, 1)
        return img, clss

    # 随机选一个数据
    def choose(self):
        return self[randint(len(self))]

    # 处理一批数据的函数
    def collate_fn(self, batch):
        _images, classes = list(zip(*batch))
        if self.transform:
            # 如果定义了数据转换操作，对每张图像进行转换操作，并添加一个维度
            images = [self.transform(img)[None] for img in _images]
        # [torch.tensor([1]), torch.tensor([0])]
        classes = [torch.tensor([id2int[clss]]) for clss in classes]
        # torch.cat(images)操作后，新张量的维度将是(n, C, H, W) classe经过cat 后 是(n,1)
        images, classes = [torch.cat(i).to(device) for i in [images, classes]]
        return images, classes, _images


all_files = Glob("cell_images/*/*.png")
np.random.seed(10)
np.random.shuffle(all_files)
# 切分训练集和测试集数据
train_files, val_files = train_test_split(all_files, random_state=1)

train_ds = MalariaImages(train_files, transform=train_transform)
val_ds = MalariaImages(val_files, transform=valid_transform)

train_dl = DataLoader(train_ds, 32, shuffle=True, collate_fn=train_ds.collate_fn)
val_dl = DataLoader(val_ds, 32, shuffle=True, collate_fn=val_ds.collate_fn)


def ConvBlock(ni, no):
    return nn.Sequential(
        nn.Dropout(0.2),
        nn.Conv2d(ni, no, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(no),  # 卷积层的批归一化操作。它的作用是对神经网络中每个批次（batch）的输入进行归一化处理，以加速训练过程并提高模型的泛化能力
        nn.MaxPool2d(2)
    )


class MalariaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
            ConvBlock(256, 512), ConvBlock(512, 64),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Linear(256, len(id2int))
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, pred, targets):
        loss = self.loss_fn(pred, targets)
        acc = (torch.max(pred, 1)[1] == targets).float().mean()
        return loss, acc


def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, labels, _ = data
    _pred = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_pred, labels)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, labels, _ = data
    _pred = model(ims)
    loss, acc = criterion(_pred, labels)
    return loss.item(), acc.item()


model = MalariaClassifier().to(device)
criterion = model.compute_metrics
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 2
# 生成日志记录器
log = Report(n_epochs)

for ex in range(n_epochs):
    N = len(train_dl)
    for bx, data, in enumerate(train_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(ex + (bx + 1) / N, trn_loss=loss, trn_acc=acc, end='\r')

    N = len(val_dl)
    for bx, data, in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        log.record(ex + (bx + 1) / N, val_loss=loss, val_acc=acc, end='\r')
    # 记录
    log.report_avgs(ex + 1)

# 下面是热力图的处理方式
# 这一步是获取模型的卷积层的模型 并使用该模型输出该图像的激活热图
im2fmap = nn.Sequential(*(list(model.model[:5].children()) + list(model.model[5][:2].children())))


def im2gradCAM(x):
    model.eval()
    logits = model(x)
    activations = im2fmap(x)
    print(activations.shape)
    # 找到logits中每行最大值的索引，即模型的预测类别。
    pred = logits.max(-1)[-1]
    # 获取模型的预测
    model.zero_grad()
    # 对logits中对应于预测类别的元素进行反向传播，计算梯度。retain_graph=True表示在反向传播后不释放计算图，
    # 这对于在同一个图中进行多次反向传播是必要的。
    logits[0, pred].backward(retain_graph=True)

    # 模型的某个特定层（这里假设是倒数第7层的第二个子层）的梯度中提取全局平均池化的梯度。这些梯度用于后续的热图生成
    # 对0，2，3维进行权重平均化
    # 512个梯度的平均值 在0，2，3维度上
    pooled_grads = model.model[-7][1].weight.grad.data.mean((0, 2, 3))

    for i in range(activations.shape[1]):
        # 将每个卷积通道的激活与对应的全局平均池化梯度相乘，得到Grad-CAM的加权激活
        activations[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(activations, dim=1)[0].cpu().detach()
    return heatmap, "Uninfected" if pred.item() else "Parasitized"


SZ = 128


def upsampleHeatmap(map, img):
    m, M = map.min(), map.max()
    map = 255 * ((map - m) / (M - m))
    map = np.uint8(map)
    map = cv2.resize(map, (SZ, SZ))
    map = cv2.applyColorMap(255 - map, cv2.COLORMAP_JET)
    map = np.uint8(map)
    map = np.uint8(map * 0.7 + img * 0.3)
    return map


N = 20
_val_dl = DataLoader(val_ds, batch_size=N, shuffle=True, collate_fn=val_ds.collate_fn)
x, y, z = next(iter(_val_dl))

for i in range(N):
    image = resize(z[i], SZ)
    heatmap, pred = im2gradCAM(x[i:i + 1])
    if pred == 'Uninfected':
        continue
    heatmap = upsampleHeatmap(heatmap, image)
    subplots([image, heatmap], nc=2, figsize=(5, 3), suptitle=pred)
