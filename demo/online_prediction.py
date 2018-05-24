import pandas as pd
import numpy as np
import sys
sys.path.append("../")
from gpcpd.kernels import GaussianKernel
from gpcpd.functions import ARGP
from gpcpd.online_gpcpd import Online_GPCPD
from matplotlib import pyplot as plt
from functools import partial

def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)

def generate_multivariate_time_series(num, dim, minl=50, maxl=1000):
    """
    Sample Data (currently not used)
    """
    data = np.empty(shape=(1, dim), dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = 10*np.random.standard_normal(dim)
        sig = np.random.standard_normal((dim,dim))
        cov = np.dot(sig, sig.T)
        
        tdata = np.random.multivariate_normal(mean, cov, p)
        data = np.concatenate((data, tdata))
    return data

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
    
    # Data Loading (Only first 4000 data)
    data = pd.read_csv("../data/processed/portfolio.csv")
    data = np.asarray(data)[:4000, 2:]
    
    # Model definition (Hyperparameter tuning needed)
    kernel = GaussianKernel(amplitude=1.0, lengthscales=10.0, n_dims=30)
    upm_func = ARGP(kernel=kernel, capacity=200, n_features=30)
    model = Online_GPCPD()
    _, maxes = model.changepoint_detection(data, upm_func, partial(constant_hazard, 100))
    
    #plt.plot(np.arange(len(maxes)), maxes)
    #plt.show()

    return
if __name__ == '__main__':
    main()
