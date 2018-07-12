# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal

class GPTS(Module):
    def __init__(self, X, Y, kernel, log_noise=0.0, window_size=50, n_features=1):
        """
        Same as exact GP (in this model, X will be the time, Y will be observations)
        Args:
            start_time:  current_time
            X:           training inputs  (time spaces)
            Y:           training targets (observations)
            kernel:      kernel
            log_noise:   noise level in log scale
            window_size: how many data will be used?
        Assume that the timestep(:=dt) is 1
        elapsed time will be used in online_prediction
        TODO:   covariance matrix precomputation
        """
        super(GPTS, self).__init__()
        self.X, self.Y, self.kernel = X, Y, kernel
        self.window_size = window_size
        self.n_features = n_features                # in the paper, denoted as \tau
        self.register_parameter(name='log_noise',
                                param=Parameter(torch.zeros(1)))

    def fit(self):
        """
        Learning hyperparameters (kernel, noise level)
        Args:
            X:  Time Spaces
            Y:  observations
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        max_epoch = 1000

        # Another convergence criteria is needed.
        print ("##### GPTS optimization start !! #####")
        print ("     optimization in progress....     ")
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            noise = torch.exp(self.log_noise)
            K = self.kernel(self.X) + torch.eye(self.X.shape[0],
                    out=self.X.new()) * noise
            L = torch.potrf(K, upper=False)
            alpha, _ = torch.gesv(self.Y, L)

            # negative log likelihood
            loss = 0.5 * torch.sum(alpha**2)         # data fitting term
            loss += torch.sum(torch.log(torch.diagonal(L, offset=0)))   # log determinant

            loss.backward()
            optimizer.step()
        print ("##### GPTS optimization end !! #####")

    def posterior_distribution(self, test_datum, window_size=None):
        noise = torch.exp(self.log_noise)
        elapsed_time = self.X.shape[0]
        if window_size is None:
            window_size = self.window_size
        X = self.X[-window_size:]
        Y = self.Y[-window_size:]
        new_X = torch.Tensor([[elapsed_time]])
        new_Y = test_datum[0].expand(1, self.n_features)

        K = self.kernel(X) + noise * torch.eye(self.window_size)
        L = torch.potrf(K, upper=False)

        Kx = self.kernel(X, new_X)
        Kxx = self.kernel(new_X)
        A, _ = torch.gesv(Kx, L)
        V, _ = torch.gesv(Y, L)

        fmean = torch.mm(A.t(), V)
        fvar = Kxx - torch.mm(A.t(), A)

        self.X = torch.cat((self.X, new_X), 0)
        self.Y = torch.cat((self.Y, new_Y), 0)

        return torch.exp(MultivariateNormal(fmean, fvar).log_prob(test_datum))

    def online_prediction(self, test_Y, window_size=None):
        """
        This function performs extrapolation beyond the elapsed time
        Just for test
        """
        noise = torch.exp(self.log_noise)
        elapsed_time = self.X.shape[0]
        T = test_Y.shape[0]

        if window_size is None:
            window_size = self.window_size

        mu = []
        var = []
        for t in range(T):
            X = self.X[-window_size:]      # Last time interval
            Y = self.Y[-window_size:]      # Last observations
            new_X = torch.Tensor([[elapsed_time + t]])
            new_Y = test_Y[t].expand(1, self.n_features)

            K = self.kernel(X) + noise * torch.eye(window_size)
            L = torch.potrf(K, upper=False)

            Kx = self.kernel(X, new_X)
            Kxx = self.kernel(new_X)
            A, _ = torch.gesv(Kx, L)
            V, _ = torch.gesv(Y, L)

            fmean = torch.mm(A.t(), V)
            fvar = Kxx - torch.mm(A.t(), A)

            fmean = np.asscalar(fmean.detach().numpy())
            fvar = np.asscalar(fvar.detach().numpy())

            mu.append(fmean)
            var.append(fvar)
            self.X = torch.cat((self.X, new_X), 0)
            self.Y = torch.cat((self.Y, new_Y), 0)

        # reset X, Y
        self.X = self.X[:elapsed_time]
        self.Y = self.Y[:elapsed_time]
        return mu, var

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
