import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from numpy import pi

def kron_prod(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .repeat(1, t2_height, t2_width)
          .view(out_height, out_width)
    )
    return expanded_t1 * tiled_t2


class IndexKernel(Module):
    def __init__(self, n_tasks, rank=1, covar_factor_bounds=(-100, 100),
            log_var_bounds=(-100, 100)):
        super(IndexKernel, self).__init__()
        self.register_parameter("covar_factor",
                nn.Parameter(torch.randn(n_tasks, rank)))#,
                #bounds=covar_factor_bounds)
        self.register_parameter("log_var",
                nn.Parameter(torch.randn(n_tasks)))#,
                #bounds=log_var_bounds)

    def forward(self, i1=None, i2=None):
        covar_matrix = self.covar_factor.matmul(
                        self.covar_factor.transpose(1,0))
        covar_matrix += self.log_var.exp().diag()
        if i1 is None and i2 is None:
            return covar_matrix
        elif i1 is None or i2 is None:
            i1 = i2 = (i1 or i2)
        output_covar = covar_matrix.index_select(0, i1.view(-1)).\
                index_select(1, i2.view(-1))
        return output_covar


class ExactMultitaskGP(Module):
    def __init__(self, X, Y, kernel, rank):
        '''
        X : N x D
        Y : N x l
        rank : rank of task cov
        '''
        super(ExactMultitaskGP, self).__init__()
        self.N, self.D = tuple(X.shape)
        assert self.N == Y.shape[0]
        self.l = Y.shape[1]
        self.X, self.Y = X, Y

        # hyperparam (kernel, noise)
        self.log_noise = Parameter(torch.zeros(self.l))
        self.input_kernel = kernel
        self.task_kernel = IndexKernel(self.l, rank=rank)

        self.cov = None
        self.inv_cov = None

    def fit(self, X=None, Y=None, lr=0.05, max_epoch=60):
        # TODO device assign model agnostic
        eye = torch.ones(self.N).diag()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        vec_Y = self.Y.t().contiguous().view(-1)
        for ep in range(max_epoch):
            optimizer.zero_grad()
            noise = kron_prod(self.log_noise.exp().diag(), eye)
            K = kron_prod(self.task_kernel(), self.input_kernel(self.X)) + noise
            L = torch.potrf(K, upper=False)
            alpha, _ = torch.gesv(vec_Y, L)
            num_dims = vec_Y.shape[0]

            # Negative log likelihood
            loss = 0.5 * torch.sum(alpha**2)
            loss += 0.5 * num_dims * torch.log(torch.tensor(2*pi))
            loss += torch.sum(torch.log(torch.diagonal(L, offset=0)))
            loss.backward()
            print ("Epoch: {}, Loss: {}".format(ep+1, loss))
            optimizer.step()

    def predict(self, test_X):
        N_ = test_X.shape[0]
        eye = torch.ones(self.N).diag()
        noise = kron_prod(self.log_noise.exp().diag(), eye)
        K = kron_prod(self.task_kernel(), self.input_kernel(self.X)) + noise
        L = torch.potrf(K, upper=False)

        Kx = kron_prod(self.task_kernel(), self.input_kernel(self.X, test_X)) # (lxN)x(lxN')
        Kxx = kron_prod(self.task_kernel(), self.input_kernel(test_X))

        A, _ = torch.gesv(Kx, L) # L-1 Kx
        Y_vec = self.Y.t().contiguous().view(-1)
        V, _ = torch.gesv(Y_vec, L) # L-1 y
        fmean = torch.mm(A.t(), V)
        fvar = Kxx - torch.mm(A.t(), A)

        assert fmean.shape[0] == N_ * self.l
        return  fmean, fvar, noise, fmean.view(self.l, -1).t(),

    def set_trainset(self, X, Y):
        assert self.D == X.shape[1]
        self.N = X.shape[0]
        assert self.N == Y.shape[0]
        self.X, self.Y = X, Y
