################################################################################################################################
##---------------------------------------------------Centralized Optimizers---------------------------------------------------##
################################################################################################################################

import numpy as np
import copy as cp
import utilities as ut

## Centralized gradient descent
def CGD(pr,learning_rate,K,theta_0):
    theta = [theta_0]  
    for k in range(K):
        theta.append( theta[-1] - learning_rate * pr.grad(theta[-1]) )
        ut.monitor('CGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

## Centralized gradient descent with momentum
def CNGD(pr,learning_rate,momentum,K,theta_0):
    theta = [theta_0]  
    theta_aux = cp.deepcopy(theta_0)
    for k in range(K):
        grad = pr.grad(theta[-1])
        theta_aux_last = cp.deepcopy(theta_aux)
        theta_aux = theta[-1] - learning_rate * grad 
        theta.append( theta_aux + momentum * ( theta_aux - theta_aux_last ) )
        ut.monitor('CNGD',k,K)
    theta_opt = theta[-1]
    F_opt = pr.F_val(theta[-1])
    return theta, theta_opt, F_opt

## Centralized stochastic gradient descent
def CSGD(pr,learning_rate,K,theta_0):
    N = pr.N
    theta = cp.deepcopy(theta_0)
    theta_epoch = [ theta_0 ]
    for k in range(K):
        idx = np.random.randint(0,N)
        grad = pr.grad(theta,idx)
        theta -= learning_rate * grad 
        if (k+1) % N == 0:
            theta_epoch.append( cp.deepcopy(theta) )
        ut.monitor('CSGD',k,K)
    return theta_epoch

## Centralized gradient descent with variance reduction using SAGA
def CSAGA(pr,learning_rate,K,theta_0):
    N = pr.N
    theta = cp.deepcopy( theta_0 )
    slots_gradient = np.zeros((N,pr.p))    
    for i in range(N):
        slots_gradient[i] = pr.grad(theta, i)
    sum_gradient = np.sum(slots_gradient,axis = 0)
    theta_epoch = [ theta_0 ]
    for k in range(K-1):
        idx = np.random.randint(0,N)
        grad = pr.grad(theta, idx)
        gradf = grad - slots_gradient[idx]
        SAGA = gradf + sum_gradient/N
        sum_gradient += gradf
        slots_gradient[idx] = cp.deepcopy(grad)
        theta -= learning_rate * SAGA
        if (k+1) % N == 0:
            theta_epoch.append( cp.deepcopy(theta) )    
        ut.monitor('CSAGA',k,K)
    return theta_epoch