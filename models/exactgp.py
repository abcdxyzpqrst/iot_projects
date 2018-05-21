import numpy as np
import torch
import math

from torch.nn import Module
from torch.nn.parameter import Parameter

class ExactGPModel(Module):
    def __init__(self, X, Y, kernel, log_noise=0.0):
        super(ExactGPModel, self).__init__()
        self.X, self.Y, self.kernel = X, Y, kernel
        self.log_noise = Parameter(torch.zeros(1))

    def fit(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        max_epoch = 60
        for epoch in range(max_epoch):
            optimizer.zero_grad()
            noise = torch.exp(self.log_noise)
            K = self.kernel(self.X) + torch.eye(self.X.shape[0]) * noise
            L = torch.potrf(K, upper=False)
            alpha, _ = torch.gesv(self.Y, L)
            num_dims = self.Y.shape[0]
        
            optimizer.zero_grad()

            # Negative log likelihood
            loss = 0.5 * torch.sum(alpha**2)
            loss = 0.5 * self.X.shape[0] * torch.log(2*math.pi)
            loss += torch.sum(torch.log(torch.diagonal(L, offset=0)))

            loss.backward()
            print ("Epoch: {}, Loss: {}".format(epoch+1, loss))
            optimizer.step()

    def predict(self, test_X):
        noise = torch.exp(self.log_noise)
        K = self.kernel(self.X) + torch.eye(self.X.shape[0]) * noise
        L = torch.potrf(K, upper=False)

        Kx = self.kernel(self.X, test_X)
        Kxx = self.kernel(test_X)
        A, _ = torch.gesv(Kx, L)
        V, _ = torch.gesv(self.Y, L)

        fmean = torch.mm(A.t(), V)
        fvar = Kxx - torch.mm(A.t(), A)

        fmean = np.squeeze(fmean.detach().numpy())
        fvar = fvar.detach().numpy()
        noise_value = noise.detach().numpy()[0]
        return fmean, fvar, noise_value
