import numpy as np
import scipy as sp
import torch
from torch.nn import Module

class torch_GPCPD(Module):
    def __init__(self, hazard_func, upm_func):
        super(torch_GPCPD, self).__init__()
        self.hazard_func = hazard_func
        self.upm_func = upm_func

    def changepoint_detection(self, observations):
        return

# inherit nn.Module?
class BOCPD_GPTS(object):
    def __init__(self, gpts_model):
        super(BOCPD_GPTS, self).__init__()
    
    def changepoint_detection(self, data, upm_func, hazard_func=None):
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
                
                if amplitude < 0:
                    amplitude *= -1
                if lengthscales < 0:
                    lengthscales *= -1
                amplitude += 1
                lengthscales += 3

                
                
            else:
                pass
             
        return posterior, maxes
