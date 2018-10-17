import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from .kernel import CNNGaussianKernel, GaussianKernel

class KDR(nn.Module):
    def __init__(self, alpha, kernel, reg_lambda=0.0):
        # alpha
        super(KDR, self).__init__()
        self.alpha = alpha
        self.kernel = kernel
        self.reg_lambda = reg_lambda
        #self.register_buffer("theta", torch.ones(self.n_kernels))

    def forward(self, X_ref, X_test, X_center=None):
        #  X_ref    (batch_size x n_ref    x k x d)
        #  X_test   (batch_size x n_test   x k x d)
        #  x_center (batch_size x n_center x k x d)
        batch_size, n_ref, k, d = X_ref.shape
        batch_size, n_test, k, d = X_test.shape

        K_test = self.kernel(X_test, X_center)  # batch x n_test x n_center
        if X_center is None:
            X_center = X_test
        # TODO CNN에 넘기는 data 수 줄이기 X_test, X_center 중복됨
        K_ref = self.kernel(X_ref, X_center)  # batch x n_ref x n_center

        n_ref, n_test = X_ref.shape[1], X_test.shape[1]
        H_hat = (self.alpha / n_ref) * torch.matmul(K_ref.transpose(1 , 2), K_ref) + \
                ((1 - self.alpha) / n_test) * torch.matmul(K_test.transpose(1, 2), K_test) # batch x n_center x n_center
        h_hat = K_ref.mean(dim=1)[:, :, None]  # batch_size x n_center x 1

        #(H + reg_lambda*I)^(-1)h
        #print("H_hat shape", H_hat.shape)
        eye = torch.eye(H_hat.shape[-1])[None, :, :].to(next(self.parameters()).device)
        theta_hat = torch.gesv(h_hat, H_hat + self.reg_lambda * eye)[0]
        theta_hat = theta_hat.detach()  # batch_size x n_center x 1
        #print(theta_hat)
        #input()
        loss = theta_hat.transpose(1, 2).matmul(H_hat).matmul(theta_hat) - \
                h_hat.transpose(1, 2).matmul(theta_hat)
        loss = torch.mean(loss.squeeze())

        g_ref = K_ref.matmul(theta_hat).squeeze(dim=-1)  # batch_size x n_ref
        g_test = K_test.matmul(theta_hat).squeeze(dim=-1) # batch_size x n_test


        alpha = self.alpha

        J = - alpha * 0.5 * torch.pow(g_ref, 2).mean(dim=-1) -\
                (1 - alpha) * 0.5 * torch.pow(g_test, 2).mean(dim=-1) + \
                g_ref.mean(dim=-1) - 0.5

        return J, loss
