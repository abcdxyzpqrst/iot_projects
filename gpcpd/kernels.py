import numpy as np

class GaussianKernel(object):
    """
    RBF Kernels with ARD Hyperparameters
    """
    def __init__(self, amplitude=1.0, lengthscales=1.0, n_dims=1, eps=1e-5):
        super(GaussianKernel, self).__init__()
        self.n_dims = n_dims
        self.eps = eps      # for numerical stability
        self.amplitude = amplitude
        self.lengthscales = [self.eps + lengthscales for _ in range(n_dims)]
    
    def __call__(self, X1, X2=None):
        """
        Args:
            X1: N x D numpy matrix
            X2: M x D numpy matrix
        Returns:
            k(X1, X2):  N x M kernel matrix
        """
        
        if X1.ndim == 1:
            X1 = np.expand_dims(X1, axis=0)
        else:
            pass

        X1 = X1 / self.lengthscales
        X1s = np.sum(X1**2, axis=1)

        if X2 is None:
            dist = -2 * np.matmul(X1, np.transpose(X1))
            dist += X1s.reshape(-1, 1) + X1s.reshape(1, -1)
            cov = self.amplitude * np.exp(-dist / 2)
            return cov

        else:
            if X2.ndim == 1:
                X2 = np.expand_dims(X2, axis=0)
            else:
                pass
            # Kernel Computation
            X2 = X2 / self.lengthscales
            X2s = np.sum(X2**2, axis=1)
            dist = -2 * np.matmul(X1, X2.T)
            dist += X1s.reshape(-1, 1) + X2s.reshape(1, -1)
            cov = self.amplitude * np.exp(-dist / 2)
            return cov
