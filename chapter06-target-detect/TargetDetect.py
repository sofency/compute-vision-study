# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/13 10:17
# Description:
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models, datasets
from torch_snippets import *
import selectivesearch
from torchvision.models import VGG16_Weights
from torchvision.ops import nms
from torch.utils.data import TensorDataset, DataLoader
import warnings
from torchsummary import summary
"""
目标检测R-CNN原理
该模型需要的训练数据为 
input: 每个样本图像中物体的边界框，
output: 预测的锚框以及预测锚框与真实锚框之间的偏差
因此该模型为多分类任务

具体工作原理: (一张图片如何加工成所需训练数据的过程，同理批量数据)
# 处理df.csv数据
# 所需的训练数据中 已经包含图片，以及图片中对应物体的边界框坐标，以及类别
# 处理数据 返回 image, [边界框坐标]，[物体类别] 图像路径

# 处理图片
1. 抽取一张图片的候选锚框，计算候选锚框与每个物体边界框的iou值
2. 如果iou小于0.3 则设置候选框为背景，否则设置为iou较大的标签 即给每个锚框设置对应的标签
3. 将数据打包成模型需要的Dataset, 即image, 裁剪后的边界框图片，图片的候选锚框，候选锚框的类别，候选锚框与边界框的偏移量
4. 整理成DataLoader  训练输入数据: 裁剪后的边界框图片,    比较loss的输出数据: 候选框的类别[转换为数字]， 候选框与边界框的偏移量
5. 使用模型进行训练， 模型主要用于预测区域建议类别，和四个边界框偏移量
"""

# 忽略 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

image_root = "../data/bus/images/images"
# 读取数据
df_raw = pd.read_csv("../data/bus/df.csv")


# 'XMin', 'XMax', 'YMin', 'YMax' 对应于图像边框的真实值


class OpenImages(Dataset):
    def __init__(self, df, image_folder=image_root):
        self.root = image_folder
        self.df = df
        self.unique_images = df["ImageID"].unique()

    def __len__(self): return len(self.unique_images)

    def __getitem__(self, item):
        image_id = self.unique_images[item]
        image_path = f'{self.root}/{image_id}.jpg'
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # 172, 256, 3 最后是通道
        h, w, _ = image.shape
        df = self.df.copy()
        # df中可能存在 一张图片多条边界框数据
        df = df[df["ImageID"] == image_id]
        # 获取锚框位置
        boxes = df['XMin,YMin,XMax,YMax'.split(',')].values
        # 扩大为原始坐标 之前归一化
        boxes = (boxes * np.array([w, h, w, h])).astype(np.uint8).tolist()
        classes = df["LabelName"].values.tolist()
        return image, boxes, classes, image_path


data = OpenImages(df_raw, image_root)


# height, width, channel
# print(im.shape)


# 抽取候选区域
def extract_candidates(img):
    # 缩放图像可以影响生成的候选区域的数量和大小。较大的缩放因子可能会产生更多的候选区域
    # 这个参数用于设置生成的候选区域的最小尺寸（以像素为单位）。在这个例子中，min_size=100 表示生成的任何候选区域的面积都必须至少为 100 像素
    img_lbl, regions = selectivesearch.selective_search(img, scale=200, min_size=100)
    # np.prod是计算数组内所有元素的乘积 是计算图片的面积
    img_area = np.prod(img.shape[:2])
    candidates = []
    added_regions = set()  # 用集合存储已添加的区域
    for r in regions:
        if r['rect'] in added_regions: continue
        if r['size'] < (0.05 * img_area): continue
        if r['size'] > (1 * img_area): continue
        # x, y, w, h = r['rect']
        candidates.append(list(r['rect']))
        added_regions.add(r['rect'])
    return candidates


# 目前的方法只能一个一个计算， 不能批量计算
def extract_iou(boxA, boxB, epsilon=1e-5):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    width = (x2 - x1)
    height = (y2 - y1)
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height
    area_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    area_combined = area_b + area_a - area_overlap
    iou = area_overlap / (area_combined + epsilon)
    return iou


# 文件路径，边界框真值，目标类别，边界框与区域建议的偏移量, 区域建议的位置，IOU
FPATHS, GTBBS, CLSS, DELTAS, ROIS, IOUS = [], [], [], [], [], []

for ix, (im, bbs, labels, fpath) in enumerate(data):
    # 测试使用
    if ix == 5: break
    H, W, _ = im.shape
    candidates = extract_candidates(im)
    # 转换真实坐标
    candidates = np.array([(x, y, x + w, y + h) for x, y, w, h in candidates])
    ious, rois, clss, deltas = [], [], [], []

    # 计算候选框与真实框的iou值 转置为 n * len(labels) 的矩阵
    ious = np.array([[extract_iou(candidate, _bb_) for candidate in candidates] for _bb_ in bbs]).T
    for jx, candidate in enumerate(candidates):
        cx, cy, cX, cY = candidate
        # 1 * len(labels)的矩阵
        candidate_iou = ious[jx]
        # 找出候选iou中类别概率最大的索引值，并给候选的每个锚框设定概率最大的那个标签
        best_iou_at = np.argmax(candidate_iou)
        best_iou = candidate_iou[best_iou_at]

        best_bb = _x, _y, _X, _Y = bbs[best_iou_at]
        # 概率超过0.3的设置为对应的标签，低于的设置为背景
        if best_iou > 0.3:
            clss.append(labels[best_iou_at])
        else:
            clss.append("background")
        # 记录候选锚框和真实锚框的误差
        delta = np.array([_x - cx, _y - cy, _X - cX, _Y - cY]) / np.array([W, H, W, H])
        deltas.append(delta)
        rois.append(candidate / np.array([W, H, W, H]))

    FPATHS.append(fpath)
    ROIS.append(rois)
    CLSS.append(clss)  # 记录候选框的类别
    DELTAS.append(deltas)
    GTBBS.append(bbs)

FPATHS = [f'{image_root}/{stem(f)}.jpg' for f in FPATHS]
targets = pd.DataFrame(flatten(CLSS), columns=['label'])
# 主要是用来将标签转换为数字
label2target = {label: t for t, label in enumerate(targets['label'].unique())}
target2label = {t: label for label, t in label2target.items()}
# 给出backward的数字
background_class = label2target['background']

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def preprocess_image(img):
    img = torch.tensor(img).permute(2, 0, 1)
    img = normalize(img)
    return img.to(device).float()


def decode(_y):
    _, preds = _y.max(-1)
    return preds


class RCNNDataset(Dataset):
    def __init__(self, fpaths, rois, labels, deltas, gtbbs):
        self.fpaths = fpaths
        self.gtbbs = gtbbs
        self.rois = rois
        self.labels = labels
        self.deltas = deltas

    def __len__(self): return len(self.fpaths)

    def __getitem__(self, item):
        # 获取到对应的图片地址
        fpath = str(self.fpaths[item])
        image = cv2.imread(fpath, cv2.COLOR_BGR2RGB)
        H, W, _ = image.shape
        sh = np.array([W, H, W, H])
        gtbbs = self.gtbbs[item]  # 图像对应的真实的锚框位置
        rois = self.rois[item]  # 候选锚框归一化的
        bbs = (np.array(rois) * sh).astype(np.uint16)  # 候选锚框真实坐标
        labels = self.labels[item]  # 记录每个图片一系列锚框的类别 字符型
        deltas = self.deltas[item]
        # 裁剪对应的锚框
        crops = [image[y:Y, x:X] for (x, y, X, Y) in bbs]
        # crops 是当前图片中锚框的数据
        return image, crops, bbs, labels, deltas, gtbbs, fpath

    @staticmethod
    def collate_fn(batch):
        inputs, rois, rixs, labels, deltas = [], [], [], [], []
        for ix in range(len(batch)):
            image, crops, image_bbs, image_labels, image_deltas, image_gt_bbs, image_fpath = batch[ix]
            # 将锚框放缩到224，224大小
            crops = [cv2.resize(crop, (224, 224)) for crop in crops]
            #
            crops = [preprocess_image(crop / 255.)[None] for crop in crops]
            inputs.extend(crops)
            labels.extend([label2target[c] for c in image_labels])  # 转换为数字
            deltas.extend(image_deltas)

        inputs = torch.cat(inputs).to(device)
        labels = torch.tensor(labels).long().to(device)
        deltas = torch.tensor(deltas).float().to(device)
        # 对应真实的锚框, 候选锚框的标签, 以及候选锚框与真实猫狂的坐标误差
        return inputs, labels, deltas


# 9/10 用于训练模型
n_train = 2
# n_train = 9 * len(FPATHS) // 10
train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])
test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

#
# show(img=train_ds[0][0])
# show(img=train_ds[0][1][0])
print(train_ds[0][2].shape)  # 候选锚框坐标 (25, 4)
print(train_ds[0][3])  # 对应锚框的标签 (25,)
print(train_ds[0][5])  # 对应物体所在的坐标 (1, 4)

# 模型的训练的数据 是图片中标记的锚框 以及锚框内图片是什么类别
train_dataloader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_dataloader = DataLoader(test_ds, batch_size=2, collate_fn=test_ds.collate_fn, drop_last=True)
#
vgg_backbone = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg_backbone.classifier = nn.Sequential()
for param in vgg_backbone.parameters():
    param.requires_grad = False

vgg_backbone.eval().to(device)


#
# summary(vgg_backbone, torch.zeros(2, 3, 224, 224))


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lmb = 10.0
        feature_dim = 25088  # 上述卷积层输出的最后维度
        self.backbone = vgg_backbone
        # 多任务  这个是输出对应的标签预测结果
        self.cls_score = nn.Linear(feature_dim, len(label2target))
        # 预测对应的锚框位置
        self.bbox = nn.Sequential(nn.Linear(feature_dim, 512), nn.ReLU(), nn.Linear(512, 4), nn.Tanh())
        # 分类任务使用交叉熵验证
        self.cel = nn.CrossEntropyLoss()
        # 预测锚框位置使用L1损失函数
        self.sl1 = nn.L1Loss()

    def forward(self, inputs):
        feat = self.backbone(inputs)
        # 预测标签
        cls_score = self.cls_score(feat)
        # 预测锚框
        bbox = self.bbox(feat)
        return cls_score, bbox

    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        # 找出不是背景的坐标
        ixs, = torch.where(labels != int(background_class))
        _deltas = _deltas[ixs]  # 预测的偏移量误差
        deltas = deltas[ixs]  # 真实的偏移量误差

        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss.detach()
        else:
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, detection_loss.detach(), regression_loss


# criterion 就是上述的calc_loss
def train_batch(inputs, model, optimizer, criterion):
    input_data, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input_data)

    total_loss, detection_loss, regression_loss = criterion(_clss, _deltas, clss, deltas)
    # 找出输出的结果是[0,1,0,1] 0表示预测错误，1表示预测正确
    accuracies = clss == decode(_clss)
    total_loss.backward()
    optimizer.step()
    return total_loss.detach(), detection_loss, regression_loss, accuracies.cpu().numpy()


@torch.no_grad()
def validate_batch(inputs, model, criterion):
    input_data, clss, deltas = inputs
    with torch.no_grad():
        model.eval()
        _clss, _deltas = model(input_data)
        total_loss, detection_loss, regression_loss = criterion(_clss, _deltas, clss, deltas)
        accuracies = clss == decode(_clss)
    return _clss, _deltas, total_loss.detach(), detection_loss, regression_loss, accuracies.cpu().numpy()


rcnn = RCNN().to(device)
criterion = rcnn.calc_loss
optimizer = torch.optim.SGD(rcnn.parameters(), lr=1e-3)
n_epochs = 5
log = Report(n_epochs)

for epoch in range(n_epochs):
    _n = len(train_dataloader)
    for ix, inputs in enumerate(train_dataloader):
        total_loss, detection_loss, regression_loss, accuracies = train_batch(inputs, rcnn, optimizer, criterion)
        pos = (epoch + (ix + 1) / _n)
        log.record(pos, trn_loss=total_loss.item(), trn_detection_loss=detection_loss,
                   trn_regression_loss=regression_loss, trn_acc=accuracies.mean(), end='\r')

    _n = len(test_dataloader)
    for ix, inputs in enumerate(test_dataloader):
        _clss, _deltas, total_loss, detection_loss, regression_loss, accuracies = validate_batch(inputs, rcnn,
                                                                                                 criterion)
        pos = (epoch + (ix + 1) / _n)
        log.record(pos, val_loss=total_loss.item(), val_detection_loss=detection_loss,
                   val_regression_loss=regression_loss, val_acc=accuracies.mean(), end='\r')

log.plot_epochs('trn_loss,val_loss'.split(","))


# 读取一张图片并预测结果
def test_predictions(filename, show_output=True):
    img = np.array(cv2.imread(filename, cv2.COLOR_BGR2RGB))
    # 抽取候选框
    candidates = extract_candidates(img)
    # 转换为左上 右下坐标形式
    candidates = [(x, y, x + w, y + h) for x, y, w, h in candidates]
    inputs = []
    for candidate in candidates:
        x, y, X, Y = candidate
        crop = cv2.resize(img[y:Y, x:X], (224, 224))
        inputs.append(preprocess_image(crop / 255.)[None])

    # input为候选的所有归一化的锚框
    inputs = torch.cat(inputs).to(device)

    with torch.no_grad():
        rcnn.eval()
        # 给每个候选框进行预测标签和损失
        probs, deltas = rcnn(inputs)
        # 转换为可能性
        probs = torch.nn.functional.softmax(probs, -1)
        # 位置，类别
        poses, class_pred_num = torch.max(probs, -1)

    candidates = np.array(candidates)
    poses, clss, probs, deltas = [tensor.detach().cpu().numpy() for tensor in [poses, class_pred_num, probs, deltas]]

    # 找出非背景的坐标
    ixs = clss != background_class
    poses, clss, probs, deltas, candidates = [tensor[ixs] for tensor in [poses, clss, probs, deltas, candidates]]

    bbs = (candidates + deltas).astype(np.uint16)
    # 使用非极大抑制消除重复的边界框，选出置信度最大的，返回置信度最高的那个下标
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(poses), 0.05)
    # bbs 现在是使用非极大抑制取得的可信索引
    poses, clss, probs, deltas, candidates, bbs = [tensor[ixs] for tensor in
                                                   [poses, clss, probs, deltas, candidates, bbs]]
    if len(ixs) == 1:
        poses, clss, probs, deltas, candidates, bbs = [tensor[None] for tensor in
                                                       [poses, clss, probs, deltas, candidates, bbs]]
    if len(poses) == 0 and not show_output:
        return (0, 0, 224, 224), 'background', 0
    if len(poses) > 0:
        best_pred = np.argmax(poses)
        best_conf = np.max(poses)
        best_bb = bbs[best_pred]
        x, y, X, Y = best_bb
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    # 展示原图像
    show(img, ax=ax[0])
    ax[0].grid(False)
    ax[0].set_title('Original image')
    if len(poses) == 0:
        ax[1].imshow(img)
        ax[1].set_title('No objects')
        plt.show()
        return
    ax[1].set_title(target2label[clss[best_pred]])
    show(img, bbs=bbs.tolist(), texts=[target2label[c] for c in clss.tolist()], ax=ax[1],
         title='predicted bounding box and class')
    plt.show()
    return (x, y, X, Y), target2label[clss[best_pred]], best_conf


print(test_predictions(test_ds[6][-1]))
