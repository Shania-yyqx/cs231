from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        std=1e-4
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.params['W1']=std*np.random.randn(input_dim,hidden_dim)
        self.params['b1']=np.zeros(hidden_dim)
        self.params['W2']=std*np.random.randn(hidden_dim,num_classes)
        self.params['b2']=np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None,reg=0.0):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        W1,b1=self.params['W1'],self.params['b1']
        W2,b2=self.params['W2'],self.params['b2']
        N,D=X.shape

        scores=None
        # 第一次输出 用relu激活函数  [N,h]
        h_output=np.maximum(0,X.dot(W1)+b1)
        scores=h_output.dot(W2)+b2   #[N,c]

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # reshape（-1，1）jiushi转换成一行
        # 第二层输出
        shift_scores=scores-np.max(scores,axis=1).reshape(-1,1)
        softmax_output=np.exp(shift_scores)/np.sum(np.exp(shift_scores),axis=1).reshape(-1,1)

        # softmax计算损失函数
        loss=-np.sum(np.log(softmax_output[range(N),list(y)]))
        loss/=N
        loss+=0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))

        
        grads={}

        # 第二层梯度计算dZ [N,c]
        sScores=softmax_output.copy()
        # softmax的梯度 就是 pk-1(yi==k)   loss'
        sScores[range(N),list(y)]-=1
        sScores/=N
        # dW[l]=dZ[l]。A[l-1].T [h,N][N,c]  [h,c]
        grads['W2']=(h_output.T).dot(sScores)+reg*W2
        # db[l]=sum(dZ) [h,1]
        grads['b2']=np.sum(sScores,axis=0)

        # 第一层梯度计算
        # y隐藏层 [h,c] w2*h=b sScores[N,c] W2[h,c] dh=[N,h]
        dh=sScores.dot(W2.T)
        # (h_output>0)是relu的求导 ，[N,h] *这个符号是每项相乘
        dh_Relu=(h_output>0)*dh  
        grads['W1']=(X.T).dot(dh_Relu)+reg*W1  #[N,h] 
        grads['b1']=np.sum(dh_Relu,axis=0)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):

        num_train=X.shape[0]
        iterations_per_epoch=max(num_train/batch_size,1) #每一轮迭代数目
        loss_history=[]
        train_acc_history=[]
        val_acc_history=[]

        for it in range(num_iters):
          X_batch=None
          Y_batch=None

          idx=np.random.choice(num_train,batch_size,replace=True)
          X_batch=X[idx]
          Y_batch=y[idx]
          loss,grads=self.loss(X_batch,y=Y_batch)
          loss_history.append(loss)

          # 参数更新
          self.params['W2']+=-learning_rate*grads['W2']
          self.params['W1']+=-learning_rate*grads['W1']
          self.params['b2']+=-learning_rate*grads['b2']
          self.params['b1']+=-learning_rate*grads['b1']

          if verbose and it%100==0: #每迭代100此打印
            print('iteration %d / %d : loss %f ' %(it,num_iters,loss))

          if it % iterations_per_epoch==0: #每一轮迭代结束
            train_acc=(self.predict(X_batch)==Y_batch).mean()
            val_acc=(self.predict(X_val)==y_val).mean()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            learning_rate*=learning_rate_decay


        return {
          'loss_history':loss_history,
          'train_acc_history':train_acc_history,
          'val_acc_history':val_acc_history
        }


    def  predict(self,X):
        y_pred=None
        h=np.maximum(0,X.dot(self.params['W1'])+self.params['b1'])
        scores=h.dot(self.params['W2'])+self.params['b2']
        y_pred=np.argmax(scores,axis=1)
        return  y_pred  
