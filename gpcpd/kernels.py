import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

class RBF(Module):
    """
    RBF kernel with ARD hyperparameters (Automatic Relevance Determination)
    """
    def __init__(self, log_amplitude=0.0, log_lengthscales=[0.0], n_dims=1, eps=1e-5):
        super(RBF, self).__init__()
        self.n_dims = n_dims
        self.eps = eps
        self.register_parameter(name="log_amplitude",
                                param=Parameter(torch.zeros(1)))
        self.register_parameter(name="log_lengthscales",
                                param=Parameter(torch.zeros(self.n_dims)))

    def forward(self, X1, X2=None):
        amplitude = torch.exp(self.log_amplitude)
        lengthscales = torch.exp(self.log_lengthscales) + self.eps

        X1 = X1 / lengthscales
        X1s = torch.sum(X1**2, dim=1)

        if X2 is None:
            dist = -2 * torch.mm(X1, X1.t())
            dist += X1s.view(-1, 1) + X1s.view(1, -1)
            cov = amplitude * torch.exp(-dist / 2)
            return cov

        else:
            X2 = X2 / lengthscales
            X2s = torch.sum(X2**2, dim=1)
            dist = -2 * torch.mm(X1, X2.t())
            dist += X1s.view(-1, 1) + X2s.view(1, -1)
            cov = amplitude * torch.exp(-dist / 2)
            return cov

class RQ_Constant(Module):
    """
    Rational Quadratic Kernel + Constant Kernel in PyTorch

    k(x_i, x_j) = c * (1 + d(x_i, x_j)^2 / 2*\alpha*\ell^2)^(-\alpha) +
    Args:
        log_amplitude:      logc, variance
        log_roughness:      smoothness or roughness
        log_lengthscale:    lengthscales
    NOT ARD hyperparameters (NOT per-dimension)
    """
    def __init__(self, log_amplitude=0.0, log_roughness=0.0, log_lengthscale=0.0, log_noise=0.0):
        super(RQ_Constant, self).__init__()
        self.register_parameter(name='RQ_log_amplitude',
                                param=Parameter(torch.Tensor([log_amplitude])))
        self.register_parameter(name='RQ_log_roughness',
                                param=Parameter(torch.Tensor([log_roughness])))
        self.register_parameter(name='RQ_log_lengthscale',
                                param=Parameter(torch.Tensor([log_lengthscale])))
        self.register_parameter(name='Const_log_noise',
                                param=Parameter(torch.Tensor([log_noise])))

    def forward(self, X1, X2=None):
        amplitude = torch.exp(self.RQ_log_amplitude)
        lengthscale = torch.exp(self.RQ_log_lengthscale)
        roughness = torch.exp(self.RQ_log_roughness)
        noise = torch.exp(self.Const_log_noise)

        sdist = self.square_dist(X1, X2)
        const = self.constant(X1, X2)
        cov = torch.ones_like(sdist) + sdist/(2*roughness*(lengthscale**2))
        cov = amplitude * (1/cov).pow(roughness)
        cov += noise * const
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

    def constant(self, X1, X2=None):
        if X2 is None:
            return torch.ones(X1.shape[0], X1.shape[0], out=X1.new())
        else:
            return torch.ones(X1.shape[0], X2.shape[0], out=X1.new())

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
