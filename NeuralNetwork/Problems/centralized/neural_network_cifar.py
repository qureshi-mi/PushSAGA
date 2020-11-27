########################################################################################################################
####-----------------------------Basic Neural Network Structure for CIFAR-10 dataset--------------------------------####
########################################################################################################################


import numpy as np
from .problem import Problem
import cifar10
import os
# from keras.datasets import cifar10

img_dim = 3073                       # set image dimension according to the dataset. (img_dim = x*y*z + 1)
n_class = 10                         # number classes in dataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    tmp = np.exp(x)
    return tmp / tmp.sum(axis=1, keepdims=True)


def softmax_loss(Y, score):
    return - np.sum(Y * np.log(score)) / Y.shape[0]

class NN(Problem):
    def __init__(self, n_agent, n_hidden=64, n_edges=None, prob=None, limited_label = False):
        self.limited_label = limited_label
        
        # Load data
        X_train, Y_train, self.X_test, self.Y_test = self.load_data()

        # Initializing variables
        self.n_hidden = n_hidden 
        self.m_mean = int(X_train.shape[0] / n_agent)
        super().__init__(n_agent, self.m_mean, (n_hidden+1) * (img_dim + n_class), n_edges=n_edges, prob=prob)
        self.n_class = n_class
        
        # Split training data into n agents
        self.X = np.array(np.split(X_train, self.n_agent, axis=0))  
        self.Y = np.array(np.split(Y_train, self.n_agent, axis=0))  

        # Keep the whole data for easier gradient and function value computation
        self.X_train = X_train
        self.Y_train = Y_train

        # Internal buffers
        self._dW = np.zeros(self.dim)                               
        self._dw = np.zeros(self.dim)                               
        self._A1 = np.zeros((self.n_hidden+1, self.m_mean*self.n_agent)) 
        self._A2 = np.zeros((self.n_class, self.m_mean*self.n_agent))    


    def load_data(self):
#         (trainX, y_train), (testX, y_test) = cifar10.load_data()         # use this if you want have keras installed
        # use this otherwise
        cifar10.maybe_download_and_extract()                               # using this to download cifar-10 dataset without keras. 
        trainX, y_train, _ = cifar10.load_training_data()
        testX, y_test, _ = cifar10.load_test_data()

########################################################################################################################
##########The user can load their custom dataset here to run the algorithms for classification of any dataset###########
######################################################################################################################## 
        
        ## Data preprocessing
        ## Reshaping the data into a single vectos
        trainX = trainX.reshape(trainX.shape[0],trainX.shape[1]*trainX.shape[2]*trainX.shape[3])
        testX = testX.reshape(testX.shape[0],testX.shape[1]*testX.shape[2]*testX.shape[3])
                
        ## Subtract mean and normalize data
        trainX -= trainX.mean()
        trainX /= np.abs(trainX).max()
        testX -= testX.mean()
        testX /= np.abs(testX).max()  
        
        ## Append '1' to every sample as a bias term
        X_train = np.append(trainX, np.ones((trainX.shape[0],1)), axis = 1)
        X_test = np.append(testX, np.ones((testX.shape[0],1)), axis = 1)
        
        if self.limited_label == True:
            permutation = np.argsort(y_train)
            X_train = X_train[permutation]
            y_train = np.sort(y_train)
        
        ## One-hot encode labels
        Y_train = np.eye(n_class)[y_train]
        Y_test = np.eye(n_class)[y_test]
        return X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy() 


    def unpack_w(self, W):
        # This function returns references
        return W[: img_dim * (self.n_hidden+1)].reshape(img_dim, self.n_hidden+1), \
                W[img_dim * (self.n_hidden+1) :].reshape(self.n_hidden+1, n_class)

    def pack_w(self, W_1, W_2):
        # This function returns a new array
        return np.append(W_1.reshape(-1), W_2.reshape(-1))

    def grad(self, w, i=None, j=None):
        # Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine.

        if i is None: # Return the full gradient
            grad, _ = self.forward_backward(self.X_train, self.Y_train, w)
        elif j is None: # Return the gradient at machine i
            grad, _ = self.forward_backward(self.X[i], self.Y[i], w)
        else: # Return the gradient of sample j at machine i
            if type(j) is np.ndarray:
                grad, _ = self.forward_backward(self.X[i, j], self.Y[i, j], w)
            else:
                grad, _ = self.forward_backward(self.X[i, [j]], self.Y[i, [j]], w)
        return grad

    def f(self, w, i=None, j=None):
        # Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.
        if i is None: # Return the function value
            return self.forward(self.X_train, self.Y_train, w)[0]
        elif j is None: # Return the function value at machine i
            return self.forward(self.X[i], self.Y[i], w)[0]
        else: # Return the function value at machine i
            if type(j) is np.ndarray:
                return self.forward(self.X[i, j], self.Y[i, j], w)[0]
            else:
                return self.forward(self.X[i, [j]], self.Y[i, [j]], w)[0]

    def forward(self, X, Y, w):
        # Forward pass
        w1, w2 = self.unpack_w(w)
        A1 = sigmoid(X.dot(w1))
        A1[:, -1] = 1
        A2 = softmax(A1.dot(w2))
        return softmax_loss(Y, A2), A1, A2


    def forward_backward(self, X, Y, w):
        # Forward pass and back propagation
        w1, w2 = self.unpack_w(w)
        loss, A1, A2 = self.forward(X, Y, w)

        dw1, dw2 = self.unpack_w(self._dw)
        dZ2 = A2 - Y
        np.dot(A1.T, dZ2, out=dw2)
        dw2 /= X.shape[0]
        dA1 = dZ2.dot(w2.T)
        dZ1 = dA1 * A1 * (1 - A1)
        np.dot(X.T, dZ1, out=dw1)
        dw1 /= X.shape[0]
        return self._dw, loss

    def accuracy(self, w):
        _, A1, A2 = self.forward(self.X_test, self.Y_test, w)
        pred = A2.argmax(axis=1)
        labels = self.Y_test.argmax(axis=1)
        return sum(pred == labels) / len(pred)

if __name__ == '__main__':
    p = NN()


