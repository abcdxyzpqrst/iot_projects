import math
import numpy as np
import scipy as sp
import torch
from torch.nn import Module
from torch.distributions.multivariate_normal import MultivariateNormal

class BOCPD_GPTS(Module):
    def __init__(self, X, Y, gpts, hazard):
        super(BOCPD_GPTS, self).__init__()
        self.X, self.Y = gpts.X, gpts.Y
        self.gpts = gpts
        self.hazard = hazard

    def changepoint_train(self):
        """
        Learning will be done for only training data
        """   
        T = self.Y.shape[0]                              # Duration
        H = self.hazard(torch.arange(1, T+1))       # Hazard values
        R = torch.zeros((T+1, T+1))
        S = torch.zeros((T, T)) 
    
        loss = torch.zeros(1)
        Z = torch.zeros((T, 1))
        predMeans = torch.zeros((T, 1))
        predMed = torch.zeros((T, 1))
        
        # we should remove all the in-place operations
        # --> 1. try clone() method
        #     2. ...
        #R[0, 0] = 1
        noise = torch.exp(self.gpts.log_noise)
        for t in range(1, T+1):
            MRC = min(self.gpts.window_size, t)

            if MRC == 1:
                X = self.X[0].expand(1, self.gpts.n_features)
                Y = self.Y[0].expand(1, self.gpts.n_features)
            
            else:
                X = self.X[t-MRC : t-1]
                Y = self.Y[t-MRC : t-1]
                
            K = self.gpts.kernel(X) + noise * torch.eye(X.shape[0])
            L = torch.potrf(K, upper=False)
                
            Kx = self.gpts.kernel(X, self.X[t-1].expand(1, self.gpts.n_features))
            Kxx = self.gpts.kernel(self.X[t-1].expand(1, self.gpts.n_features))
            A, _ = torch.gesv(Kx, L)
            V, _ = torch.gesv(Y, L)

            mu = torch.mm(A.t(), V)
            var = Kxx - torch.mm(A.t(), A)
            
            # TODO: MultivariateNormal by ourselves?
            upm = self.multivariate_normal_pdf(self.Y[t-1], mu, var)
            loss += (R[:t, t-1].clone() * upm.clone() * (1 - H[:t].clone())).sum()
            loss += (R[:t, t-1].clone() * upm.clone() * H[:t].clone()).sum()
            R[1:t+1, t] = R[:t, t-1] * upm * (1 - H[:t])
            R[0, t] = (R[:t, t-1]* upm * H[:t]).sum()
            Z[t-1] = R[:t+1, t].sum()
            
            R[:t+1, t] /= Z[t-1]
        
        print ("##### Changepoint training start !! #####")
        print ("#####    optimization in progress....    ")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        for _ in range(100):
            optimizer.zero_grad() 
            #loss = torch.sum(Z)
            loss.backward()
            print (loss)
            optimizer.step()
        print ("##### Changepoint training end !! #####")
        return

    def multivariate_normal_pdf(self, x, mu, var):
        """
        This handles only 1-dimensional case
        """
        return torch.exp(-0.5 * (torch.log(2*math.pi*var) + ((x - mu)**2)/var))

    def online_changepoint_detection(self, data, upm_func, hazard_func=None):
        """
        posterior:  probability matrix of p(r_t | x_{1:t})
        posterior[i,j] means p(r_t = i | x{1:j})

        maxes:          MAP estimate of posterior matrix
        hazard_func:    first, we fix this by constant 
        """
        N = data.shape[0]                               # number of observations
        maxes = np.zeros(N+1)
        posterior = np.zeros(shape=(N+1, N+1),          # (N+1) x (N+1) matrix 
                             dtype=np.float64)
        posterior[0, 0] = 1                             # p(r_0 = 0) = 1 (for simplicity)
        
        lcp = 0
        # here, t starts with value 0, but we refer to this as a left-shifted seq by 1.
        for t, y in enumerate(data):
            t += 1                                      # for convenience (indexing from 1) 
            upm = upm_func.pdf(np.squeeze(y), t)        # p(x_t | r_(t-1), x_t^(r))
            hazard = hazard_func(np.array(range(t)))    # p(r_t | r_(t-1))
            
            # next two lines represent message passing from previous step
            # Update the messages, there is a new change point
            posterior[0, t] = np.sum(posterior[0:t, t-1] * upm * hazard)

            # for the case, r_t = r_(t-1) + 1
            posterior[1:t+1, t] = posterior[0:t, t-1] * upm * (1 - hazard)
            
            # Renormalizing (to make probability density)
            posterior[:, t] = posterior[:, t] / np.sum(posterior[:, t])
            
            # MAP estimate
            map_estimate = posterior[:, t].argmax()
            maxes[t] = map_estimate

            if map_estimate == 0:
                print ("Changepoint occurs at {}".format(t))
                amplitude = np.random.randn()
                lengthscales = np.random.randn()
                
                """
                if amplitude < 0:
                    amplitude *= -1
                if lengthscales < 0:
                    lengthscales *= -1
                amplitude += 1
                lengthscales += 3
                """
            else:
                pass
             
        return posterior, maxes
