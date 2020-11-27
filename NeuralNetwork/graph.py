########################################################################################################################
####-----------------------------------------------------Graph------------------------------------------------------####
########################################################################################################################

## Used to generate different types of graphs

import numpy as np
from numpy import linalg as LA
import math

"""
The class of geometric graph: undirected and directed
Two nodes are connected if they are in physical proximity
"""
class Geometric_graph:
    def __init__( self, number_of_nodes ):
        self.size = number_of_nodes
    
    def undirected(self, max_distance):
        distance_nodes = np.zeros( (self.size, self.size) )      
        strongly_connected = False
        while not strongly_connected:
            ## each row is the coordinate of a node
            coordinate_nodes = np.random.uniform(0,1,(self.size,2))  
            for i in range(self.size):
                for j in range(self.size):
                    distance_nodes[i][j] \
                    = LA.norm(coordinate_nodes[i]-coordinate_nodes[j])
                    ## if distance less than max_distance then connect
            G = ( distance_nodes <= max_distance ) * 1 
            if LA.matrix_power(G,self.size-1).all() > 0:
                strongly_connected = True
        return G

    def directed(self, max_distance, percentage):
        U = self.undirected(max_distance)
        strongly_connected = False
        while not strongly_connected:
            for i in range(self.size):
                for j in range(i-1):
                    roll = np.random.uniform(0,1)
                    if U[i][j] != 0 and U[j][i] != 0 \
                    and roll < percentage:     ## with probabiltiy being directed 
                        U[i][j] = 0
            if LA.matrix_power(U,self.size-1).all() > 0:
                strongly_connected = True
        return U

"""
The class of exponential graphs: undirected and directed
As number of nodes increases exponentially, the degree at each node increases linearly.
"""
class Exponential_graph:
    def __init__(self,number_of_nodes):
        self.size = number_of_nodes
    
    def undirected(self):
        U = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            U[i][i] = 1
            hops = np.array( range( int(math.log(self.size-1,2)) + 1 ) )  
            neighbors = np.mod( i + 2 ** hops, self.size )
            for j in neighbors:
                U[i][j] = 1
                U[j][i] = 1
        return U

    def directed(self):             ## n = 2^x.
        D = np.zeros( (self.size,self.size) )
        for i in range( self.size ):
            D[i][i] = 1
            hops = np.array( range( int(math.log(self.size-1,2)) + 1 ) )  
            neighbors = np.mod( i + 2 ** hops, self.size )
            for j in neighbors:
                D[i][j] = 1
        return D

"""
This class generates all kinds of weight matrices of interest
"""
class Weight_matrix:
    def __init__(self, adjacency_matrix):      ### adjacency matrix is 0-1 np array
        self.adj = adjacency_matrix
        self.size = len(self.adj)              ### number of nodes
        self.degree = self.adj.sum(axis = 0)   ### degree vector of the graph
        
    def metroplis(self):
        M = np.zeros( (self.size,self.size) )
        for i in range(self.size):
            for j in range(self.size):
                if i !=j and self.adj[i][j] != 0:
                    M[i][j] = 1/( 2*max(self.degree[i],self.degree[j]) );
        row_sum = np.sum(M, axis = 1)
        for i in range(self.size):
            M[i][i] = 1 - row_sum[i]
        return M

    def laplacian(self,alpha):      
        L = np.diag( self.degree ) - self.adj
        WL = np.eye(self.size) - alpha * L
        return WL

    def row_stochastic(self):                  
        row_sum = np.sum(self.adj, axis = 1)
        R = np.divide(self.adj,row_sum[:,np.newaxis])       
        return R
    
    def column_stochastic(self):                ### take 0-1 adjacency matrix (numpy array) as input
        col_sum = np.sum(self.adj, axis = 0) 
        C = np.divide(self.adj,col_sum)         ### column-stochastic weight matrix
        return C
    
    def row_stoc(self):  
        num_ele = 0
        N = self.size
        while num_ele != N*N:
            mat = np.random.randint(2,size=(N, N))
            mat = np.asmatrix(mat)
            di = np.eye(N, dtype=int)
            di = np.asmatrix(di)
            A = mat + di
            A[A>1]=1 # generation of a graph
            mat_check = LA.matrix_power(A, N-1)
            mat_check[mat_check>1]=1
            num_ele = np.sum(mat_check) # to check if matrix is full
        rowsum = np.sum(A,axis=1) # for normalization
        mat_A = A/rowsum
        mat_A = np.asarray(mat_A)
        return mat_A
    
    def col_stoc(self):  
        num_ele = 0
        N = self.size
        while num_ele != N*N:
            mat = np.random.randint(2,size=(N, N))
            mat = np.asmatrix(mat)
            di = np.eye(N, dtype=int)
            di = np.asmatrix(di)
            A = mat + di
            A[A>1]=1 # generation of a graph
            mat_check = LA.matrix_power(A, N-1)
            mat_check[mat_check>1]=1
            num_ele = np.sum(mat_check) # to check if matrix is full
        colsum = np.sum(A,axis=0) # for normalization
        mat_A = A/colsum
        mat_B = np.asarray(mat_A)
        return  mat_B