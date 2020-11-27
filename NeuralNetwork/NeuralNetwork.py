########################################################################################################################
####-------------------------------------------------Neural Network-------------------------------------------------####
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Problems.my_neural_network_mnist import NN_mnist
from Problems.my_neural_network_cifar import NN_cifar
from graph import Weight_matrix, Geometric_graph
from Optimizers import DOPTIMIZER as dopt


########################################################################################################################
####----------------------------------------------MNIST Classification----------------------------------------------####
########################################################################################################################

## Generates all the plots to compare different algorithms over Geometric directed graphs using neural networks.

"""
Data processing for MNIST
"""
n = 500                                                 # number of nodes
hidden = 64                                             # number of neurons in the hidden layer   
nn_0 = NN_mnist(n, hidden, limited_label = True)        # neural network class 
m = nn_0.b                                              # number of local data samples
d = nn_0.dim

"""
Initializing variables
"""
depoch = 150
theta_0 = np.random.randn( n,d )/10
UG = Geometric_graph(n).directed(0.07, 0.03)
B = Weight_matrix(UG).column_stochastic()   
step_size = 0.5

"""
Decentralized Algorithms
"""
## SGP
theta_SGP = dopt.SGP(nn_0,B,step_size,int(depoch*m),theta_0)            
loss_SGP, acc_SGP = NN_mnist.loss_accuracy_path(nn_0, theta_SGP)
## SADDOPT     
theta_SADDOPT = dopt.SADDOPT(nn_0,B,B,step_size,int(depoch*m),theta_0) 
loss_SADDOPT, acc_SADDOPT = NN_mnist.loss_accuracy_path(nn_0, theta_SADDOPT)
## PushSAGA     
theta_PushSAGA = dopt.Push_SAGA(nn_0,B,B,step_size,int(depoch*m),theta_0)            
loss_PushSAGA, acc_PushSAGA = NN_mnist.loss_accuracy_path(nn_0, theta_PushSAGA)

"""
Save data
"""
np.savetxt('plots/MNIST_SGP_Acc.txt',acc_SGP)
np.savetxt('plots/MNIST_SGP_Loss.txt',loss_SGP)
np.savetxt('plots/MNIST_SADDOPT_Acc.txt',acc_SADDOPT)
np.savetxt('plots/MNIST_SADDOPT_Loss.txt',loss_SADDOPT)
np.savetxt('plots/MNIST_PushSAGA_Acc.txt',acc_PushSAGA)
np.savetxt('plots/MNIST_PushSAGA_Loss.txt',loss_PushSAGA)

"""
Save plot
"""
font = FontProperties()
font.set_size(23.5)

font2 = FontProperties()
font2.set_size(15)

mark_every = 15
plt.figure(1)
plt.plot(loss_SGP,'-dy', markevery = mark_every)
plt.plot(loss_SADDOPT,'->c', markevery = mark_every)
plt.plot(loss_PushSAGA,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Loss', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.legend(('SGP', 'SADDOPT','PushSAGA'), prop=font2)
plt.title('MNIST', fontproperties=font)
plt.savefig('plots/mnist_loss.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')

plt.figure(2)
plt.plot(acc_SGP,'-dy', markevery = mark_every)
plt.plot(acc_SADDOPT,'->c', markevery = mark_every)
plt.plot(acc_PushSAGA,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Accuracy', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.ylim(0.8, 0.99)
plt.legend(('SGP', 'SADDOPT','PushSAGA'), prop=font2)
plt.title('MNIST', fontproperties=font)
plt.savefig('plots/mnist_acc.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')



########################################################################################################################
####--------------------------------------------CIFAR-10 Classification---------------------------------------------####
########################################################################################################################

"""
Data processing for CIFAR
"""
nn_1 = NN_cifar(n, hidden, limited_label = True)      # neural network class 
m = nn_1.b                                            # number of local data samples
d = nn_1.dim

"""
Initializing variables
"""
depoch = 150
theta_0 = np.random.randn( n,d )/10
step_size = 0.5

"""
Decentralized Algorithms
"""
## SGP
theta_SGP = dopt.SGP(nn_1,B,step_size,int(depoch*m),theta_0)            
loss_SGP, acc_SGP = NN_cifar.loss_accuracy_path(nn_1, theta_SGP)
## SADDOPT     
theta_SADDOPT = dopt.SADDOPT(nn_1,B,B,step_size,int(depoch*m),theta_0) 
loss_SADDOPT, acc_SADDOPT = NN_cifar.loss_accuracy_path(nn_1, theta_SADDOPT)
## PushSAGA     
theta_PushSAGA = dopt.Push_SAGA(nn_1,B,B,step_size,int(depoch*m),theta_0)            
loss_PushSAGA, acc_PushSAGA = NN_cifar.loss_accuracy_path(nn_1, theta_PushSAGA)


"""
Save data
"""
np.savetxt('plots/CIFAR_SGP_Acc.txt',acc_SGP)
np.savetxt('plots/CIFAR_SGP_Loss.txt',loss_SGP)
np.savetxt('plots/CIFAR_SADDOPT_Acc.txt',acc_SADDOPT)
np.savetxt('plots/CIFAR_SADDOPT_Loss.txt',loss_SADDOPT)
np.savetxt('plots/CIFAR_PushSAGA_Acc.txt',acc_PushSAGA)
np.savetxt('plots/CIFAR_PushSAGA_Loss.txt',loss_PushSAGA)

"""
Save plot
"""
font = FontProperties()
font.set_size(23.5)

font2 = FontProperties()
font2.set_size(15)

mark_every = 15
plt.figure(3)
plt.plot(loss_SGP,'-dy', markevery = mark_every)
plt.plot(loss_SADDOPT,'->c', markevery = mark_every)
plt.plot(loss_PushSAGA,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Loss', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.legend(('SGP', 'SADDOPT','PushSAGA'), prop=font2)
plt.title('CIFAR-10', fontproperties=font)
plt.savefig('plots/cifar_loss.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')

plt.figure(4)
plt.plot(acc_SGP,'-dy', markevery = mark_every)
plt.plot(acc_SADDOPT,'->c', markevery = mark_every)
plt.plot(acc_PushSAGA,'-sr', markevery = mark_every)
plt.xlabel('Epochs', fontproperties=font)
plt.ylabel('Accuracy', fontproperties=font)
plt.tick_params(labelsize='x-large', width=3)
plt.grid(True)
plt.legend(('SGP', 'SADDOPT','PushSAGA'), prop=font2)
plt.title('CIFAR-10', fontproperties=font)
plt.savefig('plots/cifar_acc.pdf', format = 'pdf', dpi = 4000, bbox_inches='tight')