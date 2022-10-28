from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_num=X.shape[0]
    class_num=W.shape[1]
    for i in range(train_num):
      scores=X[i].dot(W)
       #为避免数值不稳定的问题，每个分值向量都减去向量中的最大值?????????
      scores-=np.max(scores)
      # 计算损失
      sum_j=np.sum(np.exp(scores))
      syi=np.exp(scores[y[i]])
      loss += -np.log(syi/sum_j)
      
      # 计算梯度下降
      for k in range(class_num):
        p_k=np.exp(scores[y[k]])/sum_j
        dW[:,k]+=(p_k-(k==y[i]))*X[i]

    
    loss/=train_num
    dW/=train_num
    loss+=0.5*reg*np.sum(W*W)
    dW+=reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # print("这里是loss",loss)
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    train_num=X.shape[0]
    scores=X.dot(W) # train_num,10的向量
    scores -= np.max(scores,axis=1,keepdims=True)   #减去一行中最大的

    sum_i = np.sum(np.exp(scores),axis=1,keepdims=True)
    p = np.exp(scores)/sum_i

    loss = np.sum(-np.log(p[np.arange(train_num),y]))  #p[range(num_train),list(y)得到的是正确分类y_i的损失值
    loss /=train_num
    loss+=0.5*reg*np.sum(W*W)

    ind = np.zeros_like(p)
    ind[np.arange(train_num), y] = 1
    dW = (X.T).dot(p - ind)   #为什么不是(p - ind) 。dot（x）???????????????????????????
    dW /= train_num
    dW += reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
