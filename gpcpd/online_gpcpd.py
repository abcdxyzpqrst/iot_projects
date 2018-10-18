import math
import time
import numpy as np
import scipy as sp
import torch
from torch.nn import Module
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import calc_f_mean_var
from torchviz import make_dot


class BOCPD_GPTS(Module):
    def __init__(self, X, Y, gpts, hazard):
        super(BOCPD_GPTS, self).__init__()
        self.X, self.Y = gpts.X, gpts.Y
        self.gpts = gpts
        self.hazard = hazard

    def getLikelihood(self):
        """
        Learning will be done for only training data
        """
        T = self.Y.shape[0]                              # Duration
        H = self.hazard(torch.arange(1, T+1))       # Hazard values
        R = torch.zeros((T+1, 1))
        S = torch.zeros((T, T))

        nlml = torch.zeros(1)
        Z = torch.zeros((T, 1))

        # we should remove all the in-place operations
        # --> 1. try clone() method
        #     2. ...
        R[0, 0] = R[0, 0] + 1
        noise = torch.exp(self.gpts.log_noise)
        for t in range(1, T+1):
            MRC = min(self.gpts.window_size, t)

            # no previous data --> GP prior
            if MRC == 1:
                upm = self.multivariate_normal_pdf(self.Y[0], torch.zeros(1), self.gpts.kernel(self.X[0].expand(1, self.gpts.n_features)))      # first datum
                tmp = torch.cat((torch.sum(R[:t, t-1] * upm * H[:t], dim=1, keepdim=True), (R[:t, t-1] * upm * (1 - H[:t])).t()), 0)
                nlml += tmp.sum()
                tmp /= tmp.sum()
                tmp = torch.cat((tmp, torch.zeros((T-t, 1))), 0)
                R = torch.cat((R, tmp), 1)

            else:
                X = self.X[t-MRC : t-1]
                Y = self.Y[t-MRC : t-1]

                # compute upm's for r_t-1 = 1 ~ t-1
                fmean, fvar = calc_f_mean_var(X, Y, self.X[t-1].expand(1, self.gpts.n_features), self.gpts.kernel, self.gpts.log_noise)
                d = MultivariateNormal(fmean, fvar)

                # r_t-1 = 0, r_t-1 = 1 ~ t-1
                upm = self.multivariate_normal_pdf(self.Y[t-1], torch.zeros(1), self.gpts.kernel(self.X[t-1].expand(1, self.gpts.n_features))).squeeze(dim=1)
                upm = torch.cat((upm, torch.exp(d.log_prob(self.Y[t-1]))), 0)
                g = make_dot(upm, self.state_dict())
                g.view()
                print(upm)
                tmp = torch.cat((torch.sum(R[:t, t-1] * upm * H[:t], dim=0, keepdim=True), (R[:t, t-1] * upm * (1 - H[:t]))), 0).view(-1, 1)
                nlml += tmp.sum()
                tmp /= tmp.sum()
                tmp = torch.cat((tmp, torch.zeros((T-t, 1))), 0)
                R = torch.cat((R, tmp), 1)
            print(R, t)
            input()
        return nlml

    def changepoint_train(self):
        print ("##### Changepoint training start !! #####")
        print ("#####    optimization in progress....    ")
        optimizer = torch.optim.Adam(self.parameters(), lr=1.0)
        for _ in range(5):
            optimizer.zero_grad()
            loss = self.getLikelihood()
            start_time = time.time()
            loss.backward()
            end_time = time.time()
            print ("Loss backward elapsed time: ", end_time - start_time)
            print (loss)
            optimizer.step()
        print ("##### Changepoint training end !! #####")
        return

    def multivariate_normal_pdf(self, x, mu, var):
        """
        This handles only 1-dimensional case
        """
        return torch.exp(-0.5 * (torch.log(2*math.pi*var) + ((x - mu)**2)/var))

    def online_changepoint_detection(self, Y, K):
        """
        posterior:  probability matrix of p(r_t | x_{1:t})
        posterior[i,j] means p(r_t = i | x{1:j})

        Y:  whole data (train + test)
        K:  window size (how much we look back?)
        """
        T = self.Y.shape[0]                              # Duration
        H = self.hazard(torch.arange(1, T+1))       # Hazard values
        R = torch.zeros((T+1, 1))
        S = torch.zeros((T, T))

        print ("Test time: ", T, " Window size: ", K)
        Z = torch.zeros((T, 1))

        # we should remove all the in-place operations
        # --> 1. try clone() method
        #     2. ...
        R[0, 0] = R[0, 0] + 1
        noise = torch.exp(self.gpts.log_noise)
        for t in range(1, T+1):
            MRC = min(K, t)

            # no previous data --> GP prior
            if MRC == 1:
                upm = self.multivariate_normal_pdf(self.Y[0], torch.zeros(1), self.gpts.kernel(self.X[0].expand(1, self.gpts.n_features)))      # first datum
                tmp = torch.cat((torch.sum(R[:t, t-1] * upm * H[:t], dim=1, keepdim=True), (R[:t, t-1] * upm * (1 - H[:t])).t()), 0)
                Z[t-1, 0] = tmp.sum()
                tmp /= tmp.sum()
                tmp = torch.cat((tmp, torch.zeros((T-t, 1))), 0)
                R = torch.cat((R, tmp), 1)

            else:
                X = self.X[t-MRC : t-1]
                Y = self.Y[t-MRC : t-1]

                # compute upm's for r_t-1 = 1 ~ t-1
                fmean, fvar = calc_f_mean_var(X, Y, self.X[t-1].expand(1, self.gpts.n_features), self.gpts.kernel, self.gpts.log_noise)
                d = MultivariateNormal(fmean, fvar)

                # r_t-1 = 0, r_t-1 = 1 ~ t-1
                upm = self.multivariate_normal_pdf(self.Y[t-1], torch.zeros(1), self.gpts.kernel(self.X[t-1].expand(1, self.gpts.n_features))).squeeze(dim=1)
                upm = torch.cat((upm, torch.exp(d.log_prob(self.Y[t-1]))), 0)

                tmp = torch.cat((torch.sum(R[:t, t-1] * upm * H[:t], dim=0, keepdim=True), (R[:t, t-1] * upm * (1 - H[:t]))), 0).view(-1, 1)
                Z[t-1, 0] = tmp.sum()
                tmp /= tmp.sum()
                tmp = torch.cat((tmp, torch.zeros((T-t, 1))), 0)
                R = torch.cat((R, tmp), 1)

        return R
