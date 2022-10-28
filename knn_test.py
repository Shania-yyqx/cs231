import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers import KNearestNeighbor


# 新建一个画图的窗口
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载数据
# 代码在 data_utils.py 中，会将data_batch_1到5的数据作为训练集，test_batch作为测试集
cifar10_dir = "cs231n\datasets\CIFAR10"
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 为了对数据有一个认识，打印出训练集和测试集的大小 
# print('Training data shape: ', X_train.shape)
# print('Training labels shape: ', y_train.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)
# Training data shape:  (50000, 32, 32, 3)
# Training labels shape:  (50000,)
# Test data shape:  (10000, 32, 32, 3)
# Test labels shape:  (10000,)

# # 一共10个类别
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# # 每个类别展示七个例子
# samples_per_class = 7
# # enumerate(classes) ===》 [(0,plane) , ...]
# for y, cls in enumerate(classes):
#     #flatnonzero(y_train == y) 返回扁平化后矩阵中非零元素的位置（index）
#     idxs = np.flatnonzero(y_train == y)
#     #np.random.choice 从idxs(只要是ndarray都可以，但必须是一维的)中随机抽取数字，并组成指定大小(size)的数组
#     #replace:True表示可以取相同数字，False表示不可以取相同数字
#     #数组p：与数组a相对应，表示取数组a中每个元素的概率，默认为选取每个元素的概率相同。
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# 为了方便 先取出部分数据进行实验
num_training = 5000
# 【0,1,2，....4999】
mask = list(range(num_training))  
X_train = X_train[mask]
y_train = y_train[mask]
num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

## 将图像数据转置成二维的
# -1代表flatten
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
# print(X_train.shape, X_test.shape)

# 创建一个分类器
classifier = KNearestNeighbor()
# 测试的数据
classifier.train(X_train, y_train)
# 把test的丢给他 计算一下距离
dists=classifier.compute_distances_two_loops(X_test)
# 预测一下k邻近  1默认最邻近
y_test_pred = classifier.predict_labels(dists, k=5)

# 检测一下错误率
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
#  k=1的时候0.274   k=3,0.272 k=5，0.278  差得不太多而且分类效果差。
print('got %d / %d correct => accuracy: %f',num_correct, num_test, accuracy)