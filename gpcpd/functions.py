import math
import numpy as np
import scipy.linalg
from scipy.stats import multivariate_normal

class ARGP(object):
    def __init__(self, kernel, log_noise=0.0, capacity=1000, n_features=10):
        """
        Autoregressive Gaussian Process Model
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

            self.holding_data[0] = last_observation
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
                self.holding_data[0:-1] = self.holding_data[1:]             # shift upwards by one
            else:
                self.curr_n_obs += 1
            self.holding_data[-1] = last_observation
          
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
