from __future__ import print_function
# 导入模块和库
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.data_utils import load_CIFAR10
from cs231n.vis_utils import visualize_grid


# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots 设置图表的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# 自动加载外部的模块
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# %load_ext autoreload
# %autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# #家在测试数据
# #  创建一个小的网络和测试数据来检查你的实现。注意我们用了随机种子来帮助实现实验的可重复性
# input_size = 4
# hidden_size = 10
# num_classes = 3
# num_inputs = 5

# def init_toy_model():
#     np.random.seed(0)
#     return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

# def init_toy_data():
#     np.random.seed(1)
#     X = 10 * np.random.randn(num_inputs, input_size)
#     y = np.array([0, 1, 2, 2, 1])
#     return X, y

# net = init_toy_model()
# X, y = init_toy_data()

# stats = net.train(X, y, X, y,
# learning_rate=1e-1, reg=5e-6,
# num_iters=100, verbose=False)

# print('Final training loss: ', stats['loss_history'][-1])


# # 画出迭代过程的损失值变化图像
# plt.plot(stats['loss_history'])
# plt.xlabel('iteration')
# plt.ylabel('training loss')
# plt.title('Training Loss history')
# plt.show()


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):

    # 加载CIFAR-10数据
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

    # 标准化数据：先求平均图像，再将每个图像都减去其平均图像
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # 将所有的图像数据都变成行的形式
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)
    return X_train, y_train, X_val, y_val, X_test, y_test


# 调用该函数以获取我们需要的数据，查看数据集的大小
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
# print('Train data shape: ', X_train.shape)
# print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', X_val.shape)
# print('Validation labels shape: ', y_val.shape)
# print('Test data shape: ', X_test.shape)
# print('Test labels shape: ', y_test.shape)


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# 训练网络
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.5, verbose=True)


# 在验证集上进行预测
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

# 
# 用提供的默认参数，在验证集上获得0.285的验证准确率。不够好。
# 错误的一个策略是绘制损失函数值以及在优化过程中训练集和验证集之间的准确性。
# 另一个策略是可视化在第一层神经网络中学到的权重。在大多数用视觉数据训练得到的神经网络中，第一层网络的权重通常会在可视化时显示出一些可见结构

# # 绘制损失值
# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epoch')
# plt.ylabel('Clasification accuracy')
# plt.show()


# 可视化网络的权重
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()
# show_net_weights(net)



best_net = None # store the best model into this 
#将最好的模型放进这里
# 使用验证集调整超参数。在best_net变量中存储最好模型的模型参数。为了帮助调试你的网络，使用类似于上面用过的可视化方法可能会有帮助; 这里的可视化结果与上面调试较差的网络中得到的可视化结果会有显著的质的差异。
# 手动调整超参数可能很有趣，但你可能会发现编写代码自动扫描可能的超参数组合会更有用，就像我们在之前的练习中做的一样。
best_val = -1
best_stats = None
learning_rates = [1e-2, 1e-3]
regularization_strengths = [0.4, 0.5, 0.6]
results = {} 
iters = 2000 #100
for lr in learning_rates:
    for rs in regularization_strengths:
        net = TwoLayerNet(input_size, hidden_size, num_classes)

        # Train the network
        stats = net.train(X_train, y_train, X_val, y_val,
                    num_iters=iters, batch_size=200,
                    learning_rate=lr, learning_rate_decay=0.95,
                    reg=rs)

        y_train_pred = net.predict(X_train)
        acc_train = np.mean(y_train == y_train_pred)
        y_val_pred = net.predict(X_val)
        acc_val = np.mean(y_val == y_val_pred)

        results[(lr, rs)] = (acc_train, acc_val)

        if best_val < acc_val:
            best_stats = stats
            best_val = acc_val
            best_net = net

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print ("lr ",lr, "reg ", reg, "train accuracy: ", train_accuracy, "val accuracy: ", val_accuracy)

print ("best validation accuracy achieved during cross-validation: ", best_val)


# lr  0.001 reg  0.5 train accuracy:  0.5302653061224489 val accuracy:  0.505
# 可视化最好的神经网络的权重
show_net_weights(best_net)

test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
