import pandas as pd
import numpy as np
import sys
import torch
sys.path.append("../")
from gpcpd.kernels import RQ
from gpcpd.functions import GPTS, StudentT, Gamma
from gpcpd.online_gpcpd import BOCPD_GPTS
from matplotlib import pyplot as plt
from functools import partial

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

    return (Ytrain, Ytest)

def main():
    """
    Random Data Changepoint Detection &
    Industry Portfolio Demo Scripts

    Hyperparameters:
        amplitude
        lengthscales (per-dimension)
        capacity (past 'capacity' values)
        hazard value
    """
    np.random.seed(777)
    
    data = np.genfromtxt('../data/processed/kyoto_60.csv', delimiter=',')
    data = data[1:, [0,120]]
    
    data = data[10000:11000]
    print (data)
    n_train = 250
    Y = np.atleast_2d(data[:, 1]).T

    n_total = Y.shape[0]
    n_test  = n_total - n_train
    
    # normalized train & test data (in standard normal distribution)
    [Ytrain, Ytest] = stdSplit(Y, n_train)
    
    # training inputs (time values from 0)
    Ttrain = np.atleast_2d(range(n_train)).T
    Ttest  = np.atleast_2d(range(n_train, n_total)).T
    
    train_inputs = torch.Tensor(Ttrain)
    test_inputs  = torch.Tensor(Ttest)
    train_targets = torch.Tensor(Ytrain)
    test_targets  = torch.Tensor(Ytest)
    
    # GPTS model hyperparameters (pre-)training for given data
    kernel = RQ()
    model = GPTS(X=train_inputs, Y=train_targets, kernel=kernel, window_size=100)
    model.pretrain()
    
    #for name, param in model.named_parameters():
    #    print (name, param.data)

    # GPTS online extrapolation for later time (just for test)
    pred_mean, pred_var = model.online_prediction(test_targets)
   
    plt.plot(np.squeeze(Ttrain), np.squeeze(Ytrain), 'r', label='Train')
    plt.plot(np.squeeze(Ttest), np.squeeze(Ytest), 'b', label='Test')
    plt.plot(np.squeeze(Ttest), pred_mean, 'black', label='Mean')
    y1 = pred_mean - 2*np.sqrt(pred_var)
    y2 = pred_mean + 2*np.sqrt(pred_var)
    plt.fill_between(np.squeeze(Ttest), y1, y2, where=y1 < y2, facecolor='lightslategrey', alpha=0.7, label='2*std')
    plt.legend(loc='best')
    plt.show()

    # BOCPD-GPTS model training & changepoint detection
    #model.changepoint_train()

if __name__ == '__main__':
    main()
