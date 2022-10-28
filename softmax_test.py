from __future__ import print_function
import random 
import numpy as np 
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.softmax import softmax_loss_naive
import time
from cs231n.classifiers import Softmax

#调制matplotlib的画图性能
# %matplotlib inline 
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots #设置默认图大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#把获取数据和数据预处理的过程封装进一个函数里
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare it for the linear classifier. These are the same steps as we used for the SVM, but condensed to a single function. 
    """
    # 加载原始CIFAR-10数据
    cifar10_dir = 'cs231n/datasets/CIFAR10'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 从数据集中取数据子集用于后面的练习
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]


    # 数据预处理：将一幅图像变成一行存在相应的矩阵里
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))


    # 标准化数据：先求平均图像，再将每个图像都减去其平均图像，这样的预处理会加速后期最优化过程中权重参数的收敛性
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image


    # 增加偏置的维度，在原矩阵后来加上一个全是1的列
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


# 调用该函数以获取我们需要的数据，然后查看数据集大小
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
# print('Train data shape: ', X_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)
# print('dev data shape: ', X_dev.shape)
# print('dev labels shape: ', y_dev.shape)


# # Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
# loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# # As a rough sanity check, our loss should be something close to -log(0.1).
# print('loss: %f' % loss)
# print('sanity check: %f' % (-np.log(0.1)))

# tic = time.time() # 函数执行前的时间，以浮点数的形式存储在tic中
# loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time() # 函数执行完毕的时间，同样是浮点数
# print('naive loss: %e computed in %fs' % (loss_naive, toc - tic)) # 打印出函数softmax_loss_naive的执行时间


# tic = time.time()
# loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic)) # 打印出函数softmax_loss_verctorized的执行时间

# # As we did for the SVM, we use the Frobenius norm to compare the two versions of the gradient.
# # 正如我们在SVM做的一样，利用弗罗贝尼乌斯范数（Frobenius norm）来比较这两个版本的梯度

# grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
# print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized)) # 打印出两个函数返回的损失值之间的差值
# print('Gradient difference: %f' % grad_difference) #打印出两个函数返回的梯度之间的距离

# 利用验证集来微调超参数（正则化强度和学习率），你应该分别使用不同的数值范围对学习率和正则化强度进行微调。如果足够细心，你应该能在验证集上实现高于0.35的分类准确率。


results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]
learning_rates = np.logspace(-10, 10, 10) 
regularization_strengths = np.logspace(-3, 6, 10) # 使用更细致的学习率和正则化强度


# 用验证集来调整学习率和正则化强度；这跟你在SVM里做的类似；把最好的Softmax分类器保存在best_softmax里
iters = 100
for lr in learning_rates:
    for rs in regularization_strengths:
        softmax = Softmax() # 函数代码在linear classifier文件里
        softmax.train(X_train, y_train, learning_rate=lr, reg=rs, num_iters=iters)

        y_train_pred = softmax.predict(X_train)
        acc_train = np.mean(y_train == y_train_pred)
        y_val_pred = softmax.predict(X_val)
        acc_val = np.mean(y_val == y_val_pred)

        results[(lr, rs)] = (acc_train, acc_val)

        if best_val < acc_val:
            best_val = acc_val
            best_softmax = softmax 

#打印结果
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)
# best validation accuracy achieved during cross-validation: 0.347000

# 在测试集上验证我们得到的最好的softmax分类器
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
# softmax on raw pixels final test set accuracy: 0.331000


# 可视化学习到的每一个类别的权重
w = best_softmax.W[:-1,:] # 出去偏置项
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # 将权重重新变成0-255中间的值
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])