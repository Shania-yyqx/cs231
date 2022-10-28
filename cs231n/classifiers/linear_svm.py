from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.  D=32*32*3+1  C=10

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]   #类数量
    num_train = X.shape[0]    #训练集数量 
    loss = 0.0
    for i in range(num_train):  
        scores = X[i].dot(W)   #[1,3072] * [3072,10]=[1,10]
        correct_class_score = scores[y[i]]  
        # 计算和其他类的分数差 
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:  #如果咋成了误差 也就是其他类别的分数 大于 正确类的分数+1
                loss += margin
                # 对于标号那一类W（一列），要
                dW[:,y[i]] += -X[i,:].T
                dW[:,j]+=X[i,:].T
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW/=num_train
    #s
    # Add regularization to the loss.
    # 正则化的目的是控制模型的学习容量,减弱过拟合的风险。
    # L2正则化，即在损失函数(越小越好)中增加权重w的平方和 ∑1/2 λw²，正常数λ是正则化系数，λ越大说明正则化强度越大。
    loss += reg * np.sum(W * W)
    
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW+=reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    train_num=X.shape[0]
    train_class=W.shape[1]
    # 预测分数
    scores=X.dot(W)
    # 正确分数的矩阵  看一下长啥样
    #这边用到list，list是一个列表（对应第几个是正确类别）实现对每一行正确分类得分的查找
    correct_Scores=scores[range(train_num),list(y)].reshape(-1,1)   #correct_class_score.shape = (num_train,1)
     #broadcast，把得分减去正确得分，加上delta，比较出最大值
    margins=np.maximum(0,scores-correct_Scores+1)
    # 把自己那一个设置成0 因为上面应该是+1了变成1
    margins[range(train_num),list(y)]=0 #
     #broadcast，把得分减去正确得分，加上delta，比较出最大值
    loss=np.sum(margins)/train_num + 0.5*reg*np.sum(W*W)    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    coeff_mat=np.zeros((train_num,train_class))  #1500*10
    #类似matlab，得到margin中大于0的索引（分类错误的），然后给coeff_mat赋值为1
    coeff_mat[margins>0]=1
    coeff_mat[range(train_num),list(y)]=0
    #给正确类加权重，有几个错误分类那就加几个，对应朴素的有几个margin>0就加几次  
    coeff_mat[range(train_num),list(y)]=-np.sum(coeff_mat,axis=1)
    #dW为训练集矩阵乘以系数矩阵
    dW=(X.T).dot(coeff_mat)
    dW = dW/train_num +reg*W   #正则化

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
