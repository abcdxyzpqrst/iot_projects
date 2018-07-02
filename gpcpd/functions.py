# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import scipy.linalg
from torch.nn import Module
from torch.nn.parameter import Parameter
from scipy.stats import multivariate_normal
from torch.distributions.gamma import Gamma
from torch.distributions.studentT import StudentT

class Logistic_H2(Module):
    """
    H(t) = h * logistic(at + b)
    Hyperparameters: [logit(h), a, b]
    -> used as a hazard function
    """
    def __init__(self, h=1.0, a=1.0, b=1.0):
        super(Logistic_H2, self).__init__()
        self.register_parameter(name='logit_h',
                                param=Parameter(torch.Tensor([h])))
        self.register_parameter(name='slope',
                                param=Parameter(torch.Tensor([a])))
        self.register_parameter(name='intercept',
                                param=Parameter(torch.Tensor([b])))

    def logistic(self, x):
        # x should be a torch.Tensor
        return 1.0 / (1.0 + torch.exp(-x))

    def forward(self, x):
        h = self.logistic(self.logit_h)
        a = self.slope
        b = self.intercept
        
        return h * self.logistic(a*x + b)

class Gamma(Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(Gamma, self).__init__()
        self.register_parameter(name='alpha',
                                param=Parameter(torch.Tensor([alpha])))
        self.register_parameter(name='beta',
                                param=Parameter(torch.Tensor([beta])))

    def forward(self, x):
        """
        probability at the value x
        """
        prob  = self.alpha * torch.log(self.beta)
        prob += (self.alpha - 1) * torch.log(x)
        prob -= self.beta * x - torch.lgamma(self.alpha)
        return torch.exp(prob)
"""
class StudentT(Module):
    Args:
        mu: location (real)
        var: variance (positive)
        df: degree of freedom (positive)
    def __init__(self, mu, var, df):
        super(StudentT, self).__init__()
        
    def forward(self, x):
        probability at value x
         
        return
"""

class GPTS(Module):
    def __init__(self, X, Y, kernel, log_noise=0.0, window_size=50, n_features=1):
        """
        Same as exact GP (in this model, X will be the time)
        Args:
            start_time:  current_time
            X:           training inputs  (time spaces)
            Y:           training targets (observations)
            kernel:      kernel
            log_noise:   noise level in log scale
            window_size: how many data will be used? 
        Assume that the timestep is 1
        """
        super(GPTS, self).__init__()
        self.X, self.Y, self.kernel = X, Y, kernel
        self.window_size = window_size
        self.n_features = n_features
        self.register_parameter(name='log_noise',
                                param=Parameter(torch.zeros(1)))
    
    def pretrain(self):
        """
        Learning hyperparameters (kernel, noise level)
        Args:
            X:  Time Spaces
            Y:  observations
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        max_epoch = 1000
        # I think, another convergence criteria is needed.
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            noise = torch.exp(self.log_noise)
            K = self.kernel(self.X) + torch.eye(self.X.shape[0]) * noise
            L = torch.potrf(K, upper=False)
            alpha, _ = torch.gesv(self.Y, L)
            
            # negative log likelihood
            loss = 0.5 * torch.sum(alpha**2)         # data fitting term
            loss += torch.sum(torch.log(torch.diagonal(L, offset=0)))   # log determinant
            
            loss.backward()
            optimizer.step()
        print ("Optimization End !!")

    def online_prediction(self, test_Y):
        """
        This function performs extrapolation beyond the elapsed time
        """
        noise = torch.exp(self.log_noise)
        elapsed_time = self.X.shape[0]
        T = test_Y.shape[0]
        
        mu = []
        var = []
        for t in range(T):
            X = self.X[-self.window_size:]      # Last time interval
            Y = self.Y[-self.window_size:]      # Last observations
            new_X = torch.Tensor([[elapsed_time + t]])
            new_Y = test_Y[t].expand(1, self.n_features)

            K = self.kernel(X) + noise * torch.eye(self.window_size)
            L = torch.potrf(K, upper=False)

            Kx = self.kernel(X, new_X)
            Kxx = self.kernel(new_X)
            A, _ = torch.gesv(Kx, L)
            V, _ = torch.gesv(Y, L)

            fmean = torch.mm(A.t(), V)
            fvar = Kxx - torch.mm(A.t(), A)

            fmean = np.asscalar(fmean.detach().numpy())
            fvar = np.asscalar(fvar.detach().numpy())
            
            self.X = torch.cat((self.X, new_X), 0)
            self.Y = torch.cat((self.Y, new_Y), 0)
        
        self.X = self.X[:elapsed_time]
        self.Y = self.Y[:elapsed_time]
        return mu, var

    def changepoint_train(self, hazard_func=None, upm_func=None):
        """
        Learning will be done for only training data
        """
        self.hazard_func = Logistic_H2()
        #self.upm_func = StudentT()
        
        T = self.Y.shape[0]                              # Duration
        H = self.hazard_func(torch.arange(1, T+1))       # Hazard values
        R = torch.zeros((T+1, T+1))
        S = torch.zeros((T, T)) 

        Z = torch.zeros((T, 1))
        predMeans = torch.zeros((T, 1))
        predMed = torch.zeros((T, 1))

        R[0, 0] = 1
        noise = torch.exp(self.log_noise)
        for t in range(1, T+1):
            print ("Time {} is being processed...".format(t))
            MRC = min(self.window_size, t)
            if MRC == 1:
                X = self.X[0].expand(1, self.n_features)
                Y = self.Y[0].expand(1, self.n_features)
            
            else:
                X = self.X[t-MRC : t-1]
                Y = self.Y[t-MRC : t-1]
                
            K = self.kernel(X) + noise * torch.eye(X.shape[0])
            L = torch.potrf(K, upper=False)
                
            Kx = self.kernel(X, self.X[t-1].expand(1, self.n_features))
            Kxx = self.kernel(self.X[t-1].expand(1, self.n_features))
            A, _ = torch.gesv(Kx, L)
            V, _ = torch.gesv(Y, L)

            mu = torch.mm(A.t(), A)
            var = torch.mm(V.t(), V)
            
            upm = torch.exp(StudentT(t+1, loc=mu, scale=var).log_prob(self.Y[t-1])) 
            R[1:t+1, t] = R[:t, t-1] * upm * (1 - H[:t])
            R[0, t] = (R[:t, t-1] * upm * H[:t]).sum()
            Z[t-1] = R[:t+1, t].sum()
            
            R[:t+1, t] /= Z[t-1]

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        for _ in range(100):
            optimizer.zero_grad()
            loss = torch.sum(Z)
            loss.backward()
            optimizer.step()
            print (loss)
        return

    def changepoint_detection(self, data, window_size):
        """
        Changepoint detection for whole dataset (train + test)
        """

        return

class ARGP(object):
    def __init__(self, kernel, log_noise=0.0, capacity=1000, n_features=10):
        """
        Autoregressive Gaussian Process Model in NumPy
        Args:
            log_noise:      noise level of log scale
            capacity:       ARGP of order p (:= capacity) -> maximum number of holding data
            n_features:     dimension of variables

            holding_data & curr_n_obs -> for AutoRegressive Gaussian Process
        """
        super(ARGP, self).__init__()
        self.kernel = kernel                # GP kernel function
        self.log_noise = log_noise          # log scale noise
        self.capacity = capacity            # maximum number of holding data
        self.n_features = n_features        # number of features
        self.holding_data = np.zeros(shape=(self.capacity, self.n_features),        # p x D design matrix (possibly not fully occupied)
                                     dtype=np.float64)
        self.curr_n_obs = 0                 # current number of holding data

    def pdf(self, last_observation):
        """
        Probability Distribution Function with GP Posterior Mean & Variance
        Args:
            last_observation:  shape=(n_features,)

        Exception:  In starting case, holding data is empty
        """
        if self.curr_n_obs == 0:
            fmean = np.zeros(self.n_features, dtype=np.float64)
            fvar = np.eye(self.n_features, dtype=np.float64)

            self.holding_data[0] = np.copy(last_observation)
            self.curr_n_obs += 1
            return multivariate_normal.pdf(last_observation, mean=fmean, cov=fvar)
        
        else:
            train_data = self.holding_data[0:self.curr_n_obs]               # shape = (p, d)
            assert train_data.shape == (self.curr_n_obs, self.n_features)
            noise = np.exp(self.log_noise)

            # Compute the GP posterior mean & variance
            K = self.kernel(train_data) + noise*np.eye(self.curr_n_obs)
            L = np.linalg.cholesky(K)                                       # K = L * L^T
            kx = self.kernel(train_data, last_observation)                  # kx = k(X, x*)
            
            A = scipy.linalg.solve_triangular(L, kx, lower=True)            # A = L^-1 * kx
            V = scipy.linalg.solve_triangular(L, train_data, lower=True)    # V = L^-1 * X
         
            fmean = np.squeeze(np.matmul(np.transpose(A), V))               # \mu = kx^T * (K + \sigma*I)^-1 * X
            fvar = self.kernel(last_observation) - np.matmul(np.transpose(A), A)
            fvar = np.eye(self.n_features) * np.asscalar(fvar)              # same variance value for each feature
            
            # we should remove oldest data (at index 0)
            if self.curr_n_obs == self.capacity:
                self.holding_data[0:-1] = np.copy(self.holding_data[1:])    # shift upwards by one
            else:
                self.curr_n_obs += 1
            self.holding_data[-1] = np.copy(last_observation)
          
            return multivariate_normal.pdf(last_observation, mean=fmean, cov=fvar)
    
    def reset_data(self):
        """
        When changepoint occurs, the data before the last changepoint should be discarded
        -> Hmm... really? (debugging point)
        """
        self.curr_n_obs = 0
        self.holding_data = np.zeros(shape=(self.capacity, self.n_features),
                                     dtype=np.float64)
        return

    def multivariate_normal_pdf(self, x, mean, cov):
        """
        Args:
            x:      observation (shape=(n_dims,))
            mean:   mean vector (shape=(n_dims,))
            cov:    covariance matrix (shape=(n_dims, n_dims))
        
            cov should be PSD matrix, so we can use Cholesky decomposition
        """
        n_dims = cov.shape[0]
        dist = x - mean
        L = np.linalg.cholesky(cov)
        V = scipy.linalg.solve_triangular(L, dist, lower=True)
        det = np.prod(np.diag(L))
        denominator = np.sqrt(np.power(2*math.pi, n_dims) * det)
        numerator   = np.exp(-1/2 * np.asscalar(np.matmul(V.T, V)))
        return numerator/denominator

    def set_hyperparameters(self, amplitude, lengthscales):
        self.kernel.set_hyperparameters(amplitude, lengthscales)

def test():
    """
    Test Scripts
    """
    observations = torch.Tensor([[0.1],
                                 [0.5],
                                 [0.4],
                                 [1.2],
                                 [1.5]])
    
    return

if __name__ == '__main__':
    test()
