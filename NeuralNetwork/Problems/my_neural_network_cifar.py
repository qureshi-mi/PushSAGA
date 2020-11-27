########################################################################################################################
####--------------------------------Decentralizing Algorithms for CIFAR-10 dataset----------------------------------####
########################################################################################################################


import numpy as np
from Problems.centralized.neural_network_cifar import NN

class NN_cifar( NN ):
    def __init__( self, n_agent, n_hidden=64, limited_label = False ):      
        super().__init__(n_agent, n_hidden, limited_label = limited_label)
        self.b = self.m_mean
        self.n = self.n_agent
        self.N_train = len(self.Y_train)
        self.data_distr = np.array([ len(_) for _ in self.X ])
    
    def localgrad(self, theta, i, j = None ): 
        ## Computes local gradient (node level)
        if j is None:                                        ## local full batch gradient
            return self.grad( theta[i], i ) 
        else:                                                ## local stochastic gradient at j
            return self.grad( theta[i], i, j ) 
        
    def networkgrad(self, theta, i_vec = None):      
        ## Computes network gradient (network level)
        ngrad = np.zeros( (self.n,self.dim) )
        if i_vec is None:                                    ## full batch gradient
            for i in range(self.n):
                ngrad[i] = self.localgrad( theta , i)
            return ngrad
        else:                                                ## component gradient: i, j
            for i in range(self.n):
                ngrad[i] = self.localgrad(theta, i, i_vec[i])
            return ngrad
    
    def F_val(self, w, i=None, j=None):             
        return self.f( w, i=None, j=None )

    def loss_accuracy_path(self, theta):
        ## computes loss and accuracy
        theta_ave = np.sum(theta, axis = 1)/self.n
        loss = [ ]
        accuracy = [ ]
        K = len(theta)
        for k in range(K):
            loss.append( self.F_val( theta_ave[k] ) )
            accuracy.append( self.accuracy( theta_ave[k] ) )
        return loss, accuracy


