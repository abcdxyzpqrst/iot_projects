import numpy as np
import torch

def offline_CPD(data, prior_function,
                observation_log_likelihood_function,
                truncate=-np.inf):
    """
    Compute the likelihood of changepoints on data.

    data                            # time-series data (in PyTorch)
    prior_function                  # a function given the likelihood of a changepoint given the distance to the last one
    observation_log_likelihood      # a function giving the log likelihood of a data part
    truncate                        # the cutoff probability 10^truncate to stop computation
                                      for that changepoint log likelihood
    P                               # the likelihoods if pre-computed
    """
    return
