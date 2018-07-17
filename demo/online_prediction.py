import pandas as pd
import numpy as np
import sys
import torch
sys.path.append("../")
sys.path.append("../gpcpd")
from graphviz import Digraph
from gpcpd.kernels import RQ_Constant
from gpcpd.functions import Gamma, Logistic_H2, Constant
from gpcpd.gpmodels import GPTS
from gpcpd.online_gpcpd import BOCPD_GPTS
from matplotlib import pyplot as plt
from functools import partial
from torchviz import make_dot, make_dot_from_trace

def stdSplit(Z, X):
    if isinstance(X, int):
        Ttrain = X
    else:
        Ttrain = X.shape[0]
    Y = np.copy(Z)

    if len(Y.shape) == 1:
        Y = np.atleast_2d(Y).T

    Ytrain = Y[:Ttrain, :]
    Ytest  = Y[Ttrain:, :]
    
    col_means = np.mean(Ytrain, axis=0)
    col_stds = np.sqrt(((Ytrain - col_means) ** 2).sum(axis=0) / (Ytrain.shape[0] - 1))

    Ytrain -= col_means[:, np.newaxis]
    Ytrain /= col_stds[:, np.newaxis]

    Ytest -= col_means[:, np.newaxis]
    Ytest /= col_stds[:, np.newaxis]

    return Ytrain, Ytest

def main():
    """
    NileRiver Level Demo in the GPCPM paper

    Hyperparameters:
        amplitude
        lengthscales (per-dimension)
        capacity (past 'capacity' values)
        hazard value
    """
    np.random.seed(777)
    torch.manual_seed(777)
    
    data = np.genfromtxt('../data/nile.txt', delimiter=',')
    #data = data[1:, [0,120]]
    
    #data = data[10000:11000]
    #print (data)
    n_train = 250
    Y = np.atleast_2d(data[:, 1]).T

    n_total = Y.shape[0]
    n_test  = n_total - n_train
    
    # normalized train & test data (in standard normal distribution)
    Ytrain, Ytest = stdSplit(Y, n_train)
    Y = np.concatenate((Ytrain, Ytest))
    
    # training inputs (time values from 0)
    Ttrain = np.atleast_2d(range(n_train)).T
    Ttest  = np.atleast_2d(range(n_train, n_total)).T
    
    train_inputs = torch.Tensor(Ttrain)
    test_inputs  = torch.Tensor(Ttest)
    train_targets = torch.Tensor(Ytrain)
    test_targets  = torch.Tensor(Ytest)
    
    # GPTS model hyperparameters (pre-)training for given data
    kernel = RQ_Constant()
    gpts = GPTS(X=train_inputs, Y=train_targets, kernel=kernel, window_size=250)
    gpts.fit()
    
    # GPTS online extrapolation for later time (just for test)
    """
    pred_mean, pred_var = model.online_prediction(test_targets)
    
    plt.plot(np.squeeze(Ttrain), np.squeeze(Ytrain), 'r', label='Train')
    plt.plot(np.squeeze(Ttest), np.squeeze(Ytest), 'b', label='Test')
    plt.plot(np.squeeze(Ttest), pred_mean, 'black', label='Mean')
    y1 = pred_mean - 2*np.sqrt(pred_var)
    y2 = pred_mean + 2*np.sqrt(pred_var)
    plt.fill_between(np.squeeze(Ttest), y1, y2, where=y1 < y2, facecolor='lightslategrey', alpha=0.7, label='2*std')
    plt.legend(loc='best')
    plt.show()
    
    # Last part
    # BOCPD-GPTS learn
    """
    hazard = Constant()
    model = BOCPD_GPTS(X=train_inputs, Y=train_targets, gpts=gpts, hazard=hazard)
    for name, param in model.named_parameters():
        print (name, param.data)

    model.changepoint_train()
    
    for name, param in model.named_parameters():
        print (name, param.data)
    
    Y = torch.Tensor(Y)
    R = model.online_changepoint_detection(Y, 500)
    g = make_dot(R, params=dict(model.named_parameters()))
    g.view()
    R = torch.argmax(R, dim=0)
    R = R.detach().numpy()
    print (R)
    
    # plotting run length posterior
    

if __name__ == '__main__':
    main()
