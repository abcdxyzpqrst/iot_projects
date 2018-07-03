import torch
from torch.nn import Module

class IncrementalCholesky(Module):
    def __init__(self, A):
        self.A = A
        self.L_A = torch.potrf(A, upper=False)
        #self.inv_L_A = self.L_A.inverse()

    def update(self, B, D):
        '''
        get cholesky form (L) of
        [ A  B.t()]
        [ B    D  ]
        when we know the L_A s.t L_A L_A.t() = A
        '''
        n = self.A.shape[0]
        m = D.shape[0]
        assert B.shape[0] == m and B.shape[1] == n

        B_ = torch.gesv(B.t(), self.L_A)[0].t()

        S = D - B_.mm(B_.t())
        try: # S is not a scalar
            L_S = torch.potrf(S, upper=False)
        except: # S is a scalar value
            L_S = torch.sqrt(S)

        self.L_A = torch.cat((
            torch.cat((self.L_A, torch.zeros(n, m)), dim=1),
            torch.cat((B_, L_S), dim=1)), dim=0)
        self.A = torch.cat((
            torch.cat((A, B.t()), dim=1),
            torch.cat((B, D), dim=1)), dim=0)

        return self.L_A, self.A

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
             else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
             for i in range(x.dim()))
    return x[inds]

def calc_f_mean_var(X_prev, Y_prev, X_curr, kernel, task_kernel):
    T = X_prev.shape[0]
    D = X_prev.shape[1]
    l = Y_prev.shape[1]
    assert T == Y_prev.shape[0]

    # reverse order
    reverse_ind = torch.arange(T-1, -1, -1).long()
    X_prev_rev = X_prev[reverse_ind]
    Y_prev_rev = Y_prev[reverse_ind]


    K_ss = kernel(X_curr)
    K_star = kernel(X_prev_rev, X_curr)
    K = kernel(X_prev_rev, X_prev_rev)

    if l > 1:
        # TODO multitask case
        pass

    cholesky = IncrementalCholesky(K[:l, :l]) # most recent
    for t in range(T):
        pass



if __name__ == '__main__':
    A = torch.FloatTensor([[4, 12], [12, 37]])
    B = torch.FloatTensor([[-16, -43]])
    D = torch.FloatTensor([[98]])

    cholesky = IncrementalCholesky(A)
    L, S = cholesky.update(B, D)

    print(L)
