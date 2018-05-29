import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

class RQ(Module):
    """
    Rational Quadratic Kernels in PyTorch
    
    k(x_i, x_j) = c * (1 + d(x_i, x_j)^2 / 2*\alpha*\ell^2)^(-\alpha)
    Args:
        log_amplitude:      logc, variance
        log_roughness:      smoothness or roughness
        log_lengthscale:    lengthscales
    NOT ARD hyperparameters (NOT per-dimension)
    """
    def __init__(self, log_amplitude=0.0, log_roughness=0.0, log_lengthscale=0.0):
        super(RQ, self).__init__()
        self.register_parameter(name='log_amplitude',
                                param=Parameter(torch.Tensor([log_amplitude])))
        self.register_parameter(name='log_roughness',
                                param=Parameter(torch.Tensor([log_roughness])))
        self.register_parameter(name='log_lengthscale',
                                param=Parameter(torch.Tensor([log_lengthscale])))

    def forward(self, X1, X2=None):
        amplitude = torch.exp(self.log_amplitude)
        lengthscale = torch.exp(self.log_lengthscale)
        roughness = torch.exp(self.log_roughness)

        sdist = self.square_dist(X1, X2)
        cov = torch.ones(sdist.shape) + sdist/(2*roughness*(lengthscale**2))
        cov = amplitude * (1/cov).pow(roughness)
        return cov

    def square_dist(self, X1, X2=None):
        """
        Params:
            X1  : N * D design matrix
            X2  : M * D design matrix

        Returns:
            N * M square dist matrix
        """
        X1s = torch.sum(X1**2, dim=1)
        
        if X2 is None:
            sdist = -2 * torch.mm(X1, X1.t())
            sdist += X1s.view(-1, 1) + X1s.view(1, -1)
            return sdist

        else:
            X2s = torch.sum(X2**2, dim=1)
            sdist = -2 * torch.mm(X1, X2.t())
            sdist += X1s.view(-1, 1) + X2s.view(1, -1)
            return sdist

class GaussianKernel(object):
    """
    RBF Kernels with ARD Hyperparameters in NumPy
    """
    def __init__(self, amplitude=1.0, lengthscales=1.0, n_dims=1, eps=1e-5):
        super(GaussianKernel, self).__init__()
        self.n_dims = n_dims
        self.eps = eps      # for numerical stability
        self.amplitude = amplitude
        self.lengthscales = [self.eps + lengthscales for _ in range(self.n_dims)]
    
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

    def set_hyperparameters(self, amplitude, lengthscales):
        self.amplitude = amplitude
        self.lengthscales = [self.eps + lengthscales for _ in range(self.n_dims)]

def test():
    """
    Test Kernels
    RQ Kernel:  pass
    """
    a = torch.Tensor([[1,2,3],
                      [2,3,4]])
    b = torch.Tensor([[3,4,5],
                      [5,6,7],
                      [2,3,4]])

    kernel = RQ(1.0, 1.0, 1.0)
    print (kernel(a,b))
    print ("RQ kernel test pass!!")

if __name__ == '__main__':
    test()
