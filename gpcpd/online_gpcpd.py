import numpy as np
import scipy as sp
import torch

# inherit nn.Module?
class Online_GPCPD(object):
    def __init__(self):
        super(Online_GPCPD, self).__init__()
    
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
        
        lcp = 0                                         # 'L'ast 'C'hange 'P'oint
        # here, t starts with value 0, but we refer to this as a left-shifted seq by 1.
        for t, x in enumerate(data):
            t += 1                                      # for convenience (indexing from 1)
            t = t - lcp                                 # reindexing from the 'L'ast 'C'hange 'P'oint
            upm = upm_func.pdf(x)                       # p(x_t | r_(t-1), x_t^(r))
            hazard = hazard_func(np.array(range(t)))    # p(r_t | r_(t-1))
            
            # next two lines represent message passing from previous step
            # for the case, r_t = 0
            posterior[0, t+lcp] = np.sum(posterior[0:t, t+lcp-1] * upm * hazard)

            # for the case, r_t = r_(t-1) + 1
            posterior[1:t+1, t+lcp] = posterior[0:t, t+lcp-1] * upm * (1 - hazard)
            
            # Renormalizing (to make probability density)
            posterior[:, t+lcp] = posterior[:, t+lcp] / np.sum(posterior[:, t+lcp])
            
            # MAP estimate
            map_estimate = posterior[:, t+lcp].argmax()
            maxes[t+lcp] = map_estimate
            
            # Changepoint occurs -> that column makes probability 1.0 on {r_t = 0}
            # -> not needed?... (debugging point)
            if map_estimate == 0:
                print ("changepoint at {}".format(t+lcp))
                posterior[:, t+lcp] = np.zeros(N+1)
                posterior[0, t+lcp] = 1.0
                lcp = t
                upm_func.reset_data()
                upm_func.holding_data[0] = x
                upm_func.curr_n_obs = 1
            
            else:
                pass
        
        return posterior, maxes
