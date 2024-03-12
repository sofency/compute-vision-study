# -*- coding: utf-8 -*-
# Author: Sofency
# Date: 2024/3/11 20:16
# Description: 利用优秀的人脸关键点检测模型检测

import face_alignment, cv2
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

# 定义人脸对齐法
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu')
input = cv2.imread("../images/Hema.JPG")
# 获取关键点的坐标
preds = fa.get_landmarks(input)[0]
print(preds.shape)

fig, ax = plt.subplots(figsize=(5, 5))
plt.imshow(cv2.cvtColor(cv2.imread("../images/Hema.JPG"), cv2.COLOR_BGR2RGB))
ax.scatter(preds[:, 0], preds[:, 1], marker='+', c='r')
plt.show()

# 三维投影
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, device='cpu')
input = cv2.imread("../images/Hema.JPG")
# 获取关键点的坐标
preds = fa.get_landmarks(input)[0]

df = pd.DataFrame(preds)
df.columns = ["x", "y", "z"]
plt.imshow(cv2.cvtColor(cv2.imread("../images/Hema.JPG"), cv2.COLOR_BGR2RGB))
ax.scatter(preds[:, 0], preds[:, 1], marker='+', c='r')
fig = px.scatter_3d(df, x='x', y='y', z='z')
fig.show()
