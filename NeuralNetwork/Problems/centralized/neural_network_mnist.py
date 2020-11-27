########################################################################################################################
####-------------------------------Basic Neural Network Structure for MNIST dataset---------------------------------####
########################################################################################################################


import numpy as np
from .problem import Problem

import os

img_dim = 785
n_class = 10

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    tmp = np.exp(x)
    return tmp / tmp.sum(axis=1, keepdims=True)


def softmax_loss(Y, score):
    return - np.sum(Y * np.log(score)) / Y.shape[0]

class NN(Problem):
    '''f(w) = 1/n \sum l_i(w), where l_i(w) is the logistic loss'''
    
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
        self.X = np.array(np.split(X_train, self.n_agent, axis=0))  # Make a new copy to make sure it is C continuous
        self.Y = np.array(np.split(Y_train, self.n_agent, axis=0))  # Make a new copy to make sure it is C continuous

        # Keep the whole data for easier gradient and function value computation
        self.X_train = X_train
        self.Y_train = Y_train

        # Internal buffers
        self._dW = np.zeros(self.dim)          
        self._dw = np.zeros(self.dim)          
        self._A1 = np.zeros((self.n_hidden+1, self.m_mean*self.n_agent))
        self._A2 = np.zeros((self.n_class, self.m_mean*self.n_agent))  


    def load_data(self):
        if os.path.exists('mnist.npz'):
            print( 'data exists' )
            data = np.load('mnist.npz', allow_pickle=True)
            X = data['X']
            y = data['y']
        else:
            print( 'downloading data' )
            from sklearn.datasets import fetch_openml
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            np.savez_compressed('mnist', X=X, y=y)
        y = y.astype('int')
        print( 'data initialized' )

        # Subtract mean and normalize data
        X -= X.mean()
        X /= np.abs(X).max()

        # Append '1' to every sample as a bias term
        X = np.append(X, np.ones((X.shape[0],1)), axis = 1)

        # Split to train & test
        n_train = 60000
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        if self.limited_label == True:
            permutation = np.argsort(y_train)
            X_train = X_train[permutation]
            y_train = np.sort(y_train)
        
        # One-hot encode labels
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


