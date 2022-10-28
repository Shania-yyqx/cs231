from __future__ import print_function
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers.linear_svm import svm_loss_naive
import time
from cs231n.gradient_check import grad_check_sparse
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers import LinearSVM
import math

plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置默认的绘图窗口大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载训练集
cifar10_dir = 'cs231n\datasets\CIFAR10'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# 这次分为训练集 验证集
num_training = 49000
num_validation = 1000
num_test = 1000
num_dev = 500

# 验证集将会是从原始的训练集中分割出来的长度为 num_validation 的数据样本点
mask = range(num_training, num_training + num_validation)
X_val = X_train[mask]
y_val = y_train[mask]

# 训练集是原始的训练集中前 num_train 个样本
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# 开发机 测试集
# 从训练集中随机抽取一小部分的数据点作为开发集
mask = np.random.choice(num_training, num_dev, replace=False)
X_dev = X_train[mask]
y_dev = y_train[mask]

# 使用前 num_test 个测试集点作为测试集
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# 数据flatten
#np.reshape(input_array, (k,-1)), 其中k为除了最后一维的维数，-1表示并不人为指定，由k和原始数据的大小来确定最后一维的长度．
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

# 首先，基于训练数据，计算图像的平均值
mean_image = np.mean(X_train, axis=0)#计算每一列特征的平均值，共32x32x3个特征
# print(mean_image.shape)
# print(mean_image[:10]) # 查看一下特征的数据
# plt.figure(figsize=(4,4))#指定画图的框图大小
# plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # 将平均值可视化出来。
# plt.show()

# 然后: 训练集和测试集图像分别减去均值#
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
X_dev -= mean_image

# 最后，在X中添加一列1作为偏置维度，这样我们在优化时候只要考虑一个权重矩阵W就可以啦．
# 结果 be like:(49000, 3073) (1000, 3073) (1000, 3073) (500, 3073)  多了一列
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])   
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

# 评估我们提供给你的loss的朴素的实现．



# 生成一个很小的SVM随机权重矩阵
# 真的很小，先标准正态随机然后乘0.0001
W = np.random.randn(3073, 10) * 0.0001 

# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)  # 从dev数据集种的样本抽样计算的loss是。。。大概估计下多少，随机几次，loss在8至9之间
# # print('loss: %f' % (loss, ))   loss： 9.363347

# # 实现梯度之后，运行下面的代码重新计算梯度．
# # 输出是grad_check_sparse函数的结果,2种情况下，可以看出，其实2种算法误差已经几乎不计了。。。
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)


# # 对随机选的几个维度计算数值梯度，并把它和你计算的解析梯度比较．所有维度应该几乎相等．
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
# grad_numerical = grad_check_sparse(f, W, grad)

# # 再次验证梯度．这次使用正则项．你肯定没有忘记正则化梯度吧~
# print('turn on reg')
# loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
# f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
# grad_numerical = grad_check_sparse(f, W, grad)

# tic = time.time()
# loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))


# tic = time.time()
# loss_vectorized, _ = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# # The losses should match but your vectorized implementation should be much faster.
# print('difference: %f' % (loss_naive - loss_vectorized))
# # Naive loss: 8.696617e+00 computed in 0.137998s
# # Vectorized loss: 8.696617e+00 computed in 0.010006s
# # difference: 0.000000

# LinearSVM是继承了classifier类，扩充了loss函数（计算梯度）
# svm = LinearSVM()
# tic = time.time()
# loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4,
#                       num_iters=1500, verbose=True)
# toc = time.time()
# print('That took %fs' % (toc - tic))

# # 看一下下降曲线
# # plt.plot(loss_hist)
# # plt.xlabel('Iteration number')
# # plt.ylabel('Loss value')
# # plt.show()
# # 编写函数 LinearSVM.predict，评估训练集和验证集的表现。
# y_train_pred = svm.predict(X_train)  
# print('training accuracy: %f' % (np.mean(y_train == y_train_pred), ))  #0.10
# y_val_pred = svm.predict(X_val)
# print('validation accuracy: %f' % (np.mean(y_val == y_val_pred), ))  #.0.08

# 使用验证集去调整超参数（正则化强度和学习率），
# 你要尝试各种不同的学习率和正则化强度
# 可以尝试先用较大的步长搜索，再微调。

learning_rates = [2e-7, 0.75e-7,1.5e-7, 1.25e-7, 0.75e-7]
regularization_strengths = [3e4, 3.25e4, 3.5e4, 3.75e4, 4e4,4.25e4, 4.5e4,4.75e4, 5e4]


# 结果是一个词典，将形式为(learning_rate, regularization_strength) 的tuples 和形式为 (training_accuracy, validation_accuracy)的tuples 对应上。准确率就简单地定义为数据集中点被正确分类的比例。

results = {}
best_val = -1   # 出现的正确率最大值
best_svm = None # 达到正确率最大值的svm对象

################################################################################
# 任务:                                                                        #
# 写下你的code ,通过验证集选择最佳超参数。对于每一个超参数的组合，
# 在训练集训练一个线性svm，在训练集和测试集上计算它的准确度，然后
# 在字典里存储这些值。另外，在 best_val 中存储最好的验证集准确度，
# 在best_svm中存储达到这个最佳值的svm对象。
#
# 提示：当你编写你的验证代码时，你应该使用较小的num_iters。这样SVM的训练模型
# 并不会花费太多的时间去训练。当你确认验证code可以正常运行之后，再用较大的
# num_iters 重跑验证代码。

################################################################################
for rate in learning_rates:
    for regular in regularization_strengths:
        svm = LinearSVM()
        svm.train(X_train, y_train, learning_rate=rate, reg=regular,   #先在训练集上训练 
                      num_iters=1000)
        y_train_pred = svm.predict(X_train)     #训练集上的准确度
        accuracy_train = np.mean(y_train == y_train_pred)   
        y_val_pred = svm.predict(X_val)    # 在测试机上测试 准确度
        accuracy_val = np.mean(y_val == y_val_pred)    
        results[(rate, regular)]=(accuracy_train, accuracy_val)  
        if (best_val < accuracy_val):   
            best_val = accuracy_val
            best_svm = svm
################################################################################
#                              结束                               #
################################################################################

for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

print ('best validation accuracy achieved during cross-validation: %f' % best_val)
# lr 1.250000e-07 reg 3.000000e+04 train accuracy: 0.377592 val accuracy: 0.393000



# x_scatter = [math.log10(x[0]) for x in results]
# y_scatter = [math.log10(x[1]) for x in results]

# #画出训练准确率
# marker_size = 100
# colors = [results[x][0] for x in results]
# plt.subplot(2, 1, 1)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 training accuracy')

# #画出验证准确率
# colors = [results[x][1] for x in results] # default size of markers is 20
# plt.subplot(2, 1, 2)
# plt.scatter(x_scatter, y_scatter, marker_size, c=colors)
# plt.colorbar()
# plt.xlabel('log learning rate')
# plt.ylabel('log regularization strength')
# plt.title('CIFAR-10 validation accuracy')
# plt.tight_layout() # 调整子图间距
# plt.show()

# y_test_pred = best_svm.predict(X_test)
# test_accuracy = np.mean(y_test == y_test_pred)
# print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

# # 在测试集上评价最好的svm的表现
# y_test_pred = best_svm.predict(X_test)
# test_accuracy = np.mean(y_test == y_test_pred)
# print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

#对于每一类，可视化学习到的权重
#依赖于你对学习权重和正则化强度的选择，这些可视化效果或者很明显或者不明显。
w = best_svm.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)
w_min, w_max = np.min(w), np.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])