import numpy as np
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

class SPGPModel(Module):
    """
    Sparse Pseudo-Inputs Gaussian Processes
    (Fully Independent Training Conditional, FITC Approximation)
    """
    def __init__(self, X, Y, kernel, Z, log_noise=0.0):
        """
        Args:
            X:  (torch.Tensor) Train Inputs (N * D design matrix)
            Y:  (torch.Tensor) Train Targets (N * 1 column vectors)
            kernel: kernels used in model
            Z:  (torch.Tensor) Initial Pseudo Inputs (M * D design matrix)
            log_noise: Initial Log Noise Value (to be learned)
        """
        super(SPGPModel, self).__init__()
        self.X, self.Y, self.kernel = X, Y, kernel
        self.log_noise = Parameter(torch.zeros(1))
        self.Z = Parameter(Z)

    def fit(self):
        Kuf = self.kernel(self.Z, self.X)
        Kuu = self.kernel(self.Z)
        
        Luu = torch.potrf(Kuu, upper=False)     # Kuu = Luu * Luu^T
        V, _ = torch.gesv(Kuf, Luu)             # V = Luu^-1 * Kuf -> V^T * V = Kfu * Kuu^-1 * Kuf
        
        diagQff = torch.sum(torch.square(V), dim=1)
        print (diagQff)
        return

    def predict(self):
        return
