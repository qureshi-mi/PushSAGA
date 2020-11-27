########################################################################################################################
####----------------------------------------Problem Class for Neural Network----------------------------------------####
########################################################################################################################


import numpy as np
import networkx as nx
## The base problem class, which generates the random problem and supports function value and gradient evaluation
class Problem(object):
    def __init__(self, n_agent, m_mean, dim, n_edges=None, prob=None, balanced=True):
        self.n_agent = n_agent      # Number of agents
        self.m_mean = m_mean        # Average number of data samples in each agent
        self.dim = dim              # Dimension of the variable
        self.X = []                 # Data
        self.Y = []                 # Noisey label
        self.Y_0 = []               # Original label
        self.w_0 = None             # The optimal value
        self.w_min = None           # The minimizer value
        self.f_min = None           # The optimal function value
        self.L = None               # The smoothness constant
        self.sigma = 0              # The strong convexity constant
        self.balanced = balanced    # Sample size is balanced over all agents or not
        self.m_total = m_mean * n_agent # Total number of data samples of all agents

        if n_edges is not None:         # Generate radom connectivity graph
            self.generate_erdos_renyi_graph(2*self.n_agent / self.n_agent / (self.n_agent-1))
        elif prob is not None:
            self.generate_erdos_renyi_graph(prob)

        # Split data
        if self.balanced == True:
            self.m = np.ones(self.n_agent, dtype=int) * self.m_mean
        else:
            tmp = np.random.random(self.n_agent)
            tmp *= self.m_total * 0.3 / tmp.sum()
            tmp = tmp.astype(int) + int(self.m_mean * 0.7)

            extra = self.m_total - tmp.sum()
            i = 0
            while extra > 0:
                tmp[i] += 1
                extra -= 1
                i += 1
                i %= self.n_agent

            self.m = tmp


    def split_data(self, m, X):
        # Helper function to split data according to the number of training samples per agent.
        cumsum = m.cumsum().astype(int).tolist()
        inds = zip([0] + cumsum[:-1], cumsum)
        return [ X[start:end] for (start, end) in inds ]


    def grad(self, w, i=None, j=None):
        '''Gradient at w. If i is None, returns the full gradient; if i is not None but j is, returns the gradient in the i-th machine; otherwise,return the gradient of j-th sample in i-th machine.'''
        pass

    def grad_vec(self, w):
        '''Gradient at w. Returns returns the matrix of gradients of all subproblems.'''
        pass

    def hessian(self, w, i=None, j=None):
        '''Hessian matrix at w. If i is None, returns the full Hessian matrix; if i is not None but j is, returns the hessian matrix in the i-th machine; otherwise,return the hessian matrix of j-th sample in i-th machine.'''
        pass

    def f(self, w, i=None, j=None):
        '''Function value at w. If i is None, returns f(x); if i is not None but j is, returns the function value in the i-th machine; otherwise,return the function value of j-th sample in i-th machine.'''
        pass


    def grad_check(self):
        # Check the gradient implementation at random variable values.
        w = np.random.rand(self.dim)
        delta = np.zeros(self.dim)
        grad = np.zeros(self.dim)
        eps = 1e-4

        for i in range(self.dim):
            delta[i] = eps
            grad[i] = (self.f(w+delta) - self.f(w-delta)) / 2 / eps
            delta[i] = 0

        if np.linalg.norm(grad - self.grad(w)) > eps:
            print('Grad check failed!')
            return False
        else:
            print('Grad check succeeded!')
            return True

    def distributed_check(self):
        # Check the function value and gradient implementations are correct.

        w = np.random.randn(self.dim)
        g = 0
        g_ij = 0
        for i in range(self.n_agent):
            g += self.grad(w, i) * self.m_mean
            for j in range(self.m[i]):
                g_ij += self.grad(w, i, j)

        g /= self.m_total
        g_ij /= self.m_total
        # print('g - 1/n \sum g_i = ' + str(np.linalg.norm(g - self.grad(w))))
        # print('g - 1/nm \sum g_ij = ' + str(np.linalg.norm(g_ij - self.grad(w))))
        if np.linalg.norm(g - self.grad(w)) > 1e-5:
            print('Distributed g check failed!')
            return False
        if np.linalg.norm(g_ij - self.grad(w)) > 1e-5:
            print('Distributed g_ij check failed!')
            return False

        f = 0
        f_ij = 0
        for i in range(self.n_agent):
            f += self.f(w, i) * self.m_mean
            for j in range(self.m[i]):
                f_ij += self.f(w, i, j)

        f /= self.m_total
        f_ij /= self.m_total
        # print('f - 1/n sum f_i = ' + str(np.abs(f - self.f(w))))
        # print('f - 1/nm sum f_ij = ' + str(np.abs(f_ij - self.f(w))))
        if np.abs(f - self.f(w)) > 1e-10:
            print(np.abs(f - self.f(w)))
            print('Distributed f check failed!')
            return False
        if np.abs(f_ij - self.f(w)) > 1e-10:
            print(np.abs(f_ij - self.f(w)))
            print('Distributed f_ij check failed!')
            return False
        print('Distributed check succeeded!')
        return True

    def generate_erdos_renyi_graph(self, prob):
        # Generate connected connectivity graph according to the params.

        if prob < 2 / (self.n_agent - 1):
            print("Need higher probability to create a connected graph!")
            exit(-1)
        G = nx.erdos_renyi_graph(self.n_agent, prob)
        if nx.is_connected(G):
            # Update number of edges of the actual graph 
            self.n_edges = G.number_of_edges()
            self.G = G
        else:
            self.generate_erdos_renyi_graph(prob)

    def generate_ring_graph(self):
        # Generate ring connectivity graph.
        G = nx.cycle_graph(self.n_agent)
        # Update number of edges of the actual graph 
        self.n_edges = G.number_of_edges()
        self.G = G

    def generate_grid_graph(self, m, n):
        # Generate m x n grid connectivity graph.
        G = nx.grid_2d_graph(m, n)
        # Update number of edges of the actual graph 
        self.n_edges = G.number_of_edges()
        self.G = G

    def generate_star_graph(self):
        # Generate m x n grid connectivity graph.
        G = nx.star_graph(self.n_agent-1)
        # Update number of edges of the actual graph 
        self.n_edges = G.number_of_edges()
        self.G = G
