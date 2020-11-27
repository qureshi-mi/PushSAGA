########################################################################################################################
####------------------------------------Logistic Regression for CIFAR-10 dataset------------------------------------####
########################################################################################################################

## Used to help implement decentralized algorithms to classify CIFAR-10 dataset

import numpy as np
from numpy import linalg as LA
import os
import sys
import cifar10
# from keras.datasets import cifar10

# Labels of different classes to select from:
# airplane = 0, automobile = 1, bird = 2, cat = 3, deer = 4, dog = 5, frog = 6, horse = 7, ship = 8, truck = 9;

class LR_L4( object ):
    def __init__(self, n_agent, class1 = 0, class2 = 1, balanced = True, limited_labels = False ):
        self.class1 = class1
        self.class2 = class2
        self.limited_labels = limited_labels
        self.n = n_agent 
        self.balanced = balanced
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.load_data()
        self.N = len(self.X_train)            ## total number of data samples
        if balanced == False:
            self.split_vec = np.sort(np.random.choice(np.arange(1,self.N),self.n-1, replace = False )) 
        self.X, self.Y, self.data_distr = self.distribute_data()
        self.p = len(self.X_train[0])         ## dimension of the feature 
        self.reg = 1/self.N
        self.dim = self.p                     ## dimension of the feature 
        self.L, self.kappa = self.smooth_scvx_parameters()
        self.b = int(self.N/self.n)           ## average local samples

    def load_data(self):
#         (trainX, trainY), (testX, testY) = cifar10.load_data()         # use this if you want have keras installed
        # use this otherwise
        cifar10.maybe_download_and_extract()                             # using this to download cifar-10 dataset without keras. 
        trainX, trainY, _ = cifar10.load_training_data()
        testX, testY, _ = cifar10.load_test_data()

########################################################################################################################
##########The user can load their custom dataset here to run the algorithms for classification of any dataset###########
########################################################################################################################
        
        ## data preprocessing
        trainX = trainX.reshape(trainX.shape[0],trainX.shape[1]*trainX.shape[2]*trainX.shape[3])
        testX = testX.reshape(testX.shape[0],testX.shape[1]*testX.shape[2]*testX.shape[3])
        
        ## append 1 to the end of all data points
        trainX = np.append(trainX, np.ones((trainX.shape[0],1)), axis = 1)
        testX = np.append(testX, np.ones((testX.shape[0],1)), axis = 1)
        
        ## data normalization: each data is normalized as a unit vector 
        trainX = trainX / LA.norm(trainX,axis = 1)[:,None]
        testX = testX / LA.norm(testX,axis = 1)[:,None]
        
        ## select corresponding classes
        trainX_C1_C2 = trainX[ (trainY == self.class1) | (trainY == self.class2) ]
        trainy_C1_C2 = trainY[ (trainY == self.class1) | (trainY == self.class2) ]
        trainy_C1_C2[ trainy_C1_C2 == self.class1 ] = 1    
        trainy_C1_C2[ trainy_C1_C2 == self.class2 ] = -1
        
        ## do the same for test data
        testX_C1_C2 = testX[ (testY == self.class1) | (testY == self.class2) ]
        testy_C1_C2 = testY[ (testY == self.class1) | (testY == self.class2) ]
        testy_C1_C2[ testy_C1_C2 == self.class1 ] = 1    
        testy_C1_C2[ testy_C1_C2 == self.class2 ] = -1

        X_train, X_test = trainX_C1_C2, testX_C1_C2
        Y_train, Y_test = trainy_C1_C2, testy_C1_C2        
        
        
        if self.limited_labels == True:
            permutation = np.argsort(Y_train)
            X_train = X_train[permutation]
            Y_train = np.sort(Y_train)
            
        return X_train.copy(), Y_train.copy(), X_test.copy(), Y_test.copy() 
    
    def distribute_data(self):
        if self.balanced == True:
           X = np.array( np.split( self.X_train, self.n, axis = 0 ) ) 
           Y = np.array( np.split( self.Y_train, self.n, axis = 0 ) ) 
        if self.balanced == False:   ## random distribution
           X = np.array( np.split(self.X_train, self.split_vec, axis = 0) )
           Y = np.array( np.split(self.Y_train, self.split_vec, axis = 0 ) )
        data_distribution = np.array([ len(_) for _ in X ])
        return X, Y, data_distribution
    
    def smooth_scvx_parameters(self):
        Q = np.matmul(self.X_train.T,self.X_train)/self.N
        L_F = max(abs(LA.eigvals(Q)))/4
        L = L_F + self.reg
        kappa = L/self.reg
        return L, kappa
    
    def F_val(self, theta):           ##  objective function value at theta
        if self.balanced == True:
            f_val = np.sum( np.log( np.exp( np.multiply(-self.Y_train,\
                                                    np.matmul(self.X_train,theta)) ) + 1 ) )/self.N
            reg_val = (self.reg/2) * (LA.norm(theta) ** 2) 
            return f_val + reg_val
        if self.balanced == False:
            temp1 = np.log( np.exp( np.multiply(-self.Y_train,\
                              np.matmul(self.X_train,theta)) ) + 1 ) 
            temp2 = np.split(temp1, self.split_vec)
            f_val = 0
            for i in range(self.n):
                f_val += np.sum(temp2[i])/self.data_distr[i]
            reg_val = (self.reg/2) * (LA.norm(theta) ** 2) 
            return f_val/self.n + reg_val
            
        
    def localgrad(self, theta, idx, j = None ):  ## idx is the node index, j is local sample index
        if j == None:                 ## local full batch gradient
            temp1 = np.exp( np.matmul(self.X[idx],theta[idx]) * (-self.Y[idx])  )
            temp2 = ( temp1/(temp1+1) ) * (-self.Y[idx])
            grad = self.X[idx] * temp2[:,np.newaxis]
            return np.sum(grad, axis = 0)/self.data_distr[idx] + self.reg * theta[idx]
        else:                         ## local stochastic gradient  
            temp = np.exp(self.Y[idx][j]*np.inner(self.X[idx][j], theta[idx]))
            grad_lr = -self.Y[idx][j]/(1+temp) * self.X[idx][j]
            grad_reg = self.reg * theta[idx]
            grad = grad_lr + grad_reg
            return grad
        
    def networkgrad(self, theta, idxv = None):  ## network stochastic/batch gradient
        grad = np.zeros( (self.n,self.p) )
        if idxv is None:                        ## full batch
            for i in range(self.n):
                grad[i] = self.localgrad(theta , i)
            return grad
        else:                                   ## stochastic gradient: one sample
            for i in range(self.n):
                grad[i] = self.localgrad(theta, i, idxv[i])
            return grad
    
    def grad(self, theta, idx = None): ## centralized stochastic/batch gradient
        if idx == None:                ## full batch
            if self.balanced == True:
                temp1 = np.exp( np.matmul(self.X_train,theta) * (-self.Y_train)  )
                temp2 = ( temp1/(temp1+1) ) * (-self.Y_train)
                grad = self.X_train * temp2[:,np.newaxis]
                return np.sum(grad, axis = 0)/self.N + self.reg * theta
            if self.balanced == False:
                return np.sum( self.networkgrad(np.tile(theta,(self.n,1)))\
                              , axis = 0 )/self.n
        else:
            if self.balanced == True:
                temp = np.exp(self.Y_train[idx]*np.inner(self.X_train[idx], theta))
                grad_lr = -self.Y_train[idx]/(1+temp) * self.X_train[idx]
                grad_reg = self.reg * theta
                grad = grad_lr + grad_reg
                return grad
            if self.balanced == False:
                sys.exit( 'data distribution is not balanced !!!' )
    
