# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import scipy.linalg
from torch.nn import Module
from torch.nn.parameter import Parameter
from scipy.stats import multivariate_normal
from torch.distributions.gamma import Gamma
from torch.distributions.studentT import StudentT

class Constant(Module):
    """
    TODO:   implement the constant hazard function
    """
    pass

class Logistic_H2(Module):
    """
    H(t) = h * logistic(at + b)
    Hyperparameters: [logit(h), a, b]
    -> used as a hazard function
    """
    def __init__(self, h=1.0, a=1.0, b=1.0):
        super(Logistic_H2, self).__init__()
        self.register_parameter(name='logit_h',
                                param=Parameter(torch.Tensor([h])))
        self.register_parameter(name='slope',
                                param=Parameter(torch.Tensor([a])))
        self.register_parameter(name='intercept',
                                param=Parameter(torch.Tensor([b])))

    def logistic(self, x):
        # x should be a torch.Tensor
        return 1.0 / (1.0 + torch.exp(-x))

    def forward(self, x):
        h = self.logistic(self.logit_h)
        a = self.slope
        b = self.intercept
        return h * self.logistic(a*x + b)

class Log_Gamma(Module):
    def __init__(self, alpha=2.0, beta=2.0):
        super(Log_Gamma, self).__init__()
        self.register_parameter(name='alpha',
                                param=Parameter(torch.Tensor([alpha])))
        self.register_parameter(name='beta',
                                param=Parameter(torch.Tensor([beta])))

    def forward(self, x):
        """
        log probability at the value x
        """
        log_prob  = self.alpha * torch.log(self.beta)
        log_prob += (self.alpha - 1) * torch.log(x)
        log_prob -= self.beta * x - torch.lgamma(self.alpha)
        return log_prob

class StudentT(Module):
    def __init__(self):
        super(StudentT, self).__init__()

    def forward(self, x):
        return

def test():
    """
    Test Scripts
    """
    log_gamma = Log_Gamma()
    print (log_gamma(torch.Tensor([2.0])))
    return

if __name__ == '__main__':
    test()
