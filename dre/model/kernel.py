#!/usr/bin/env python
# -*- c-file-style: "sourcery" -*-
#
# Use and distribution of this software and its source code is governed
# by the terms and conditions defined in the "LICENSE" file that is part
# of this source code package.
#
# copy and paste from https://github.com/paolodedios/shift-detect
#
# TODO change to pytorch version!!

"""
Kernel interfaces
"""

from __future__             import print_function
import numpy     as np
import torch
import torch.functional as F
import torch.nn as nn
from .cnn1d import CNN

class GaussianKernel(nn.Module):
    def __init__(self, length_scale=1.0):
        super(GaussianKernel, self).__init__()
        self.length_scale = length_scale

    def compute_distance(self, X1, X2): #samples=None, sampleMeans=None) :a
        """
        Compute the distances between points in the sample's feature space
        to points along the center of the distribution
        """
        batch_size = X1.shape[0]
        n1, n2 = X1.shape[1], X2.shape[1]
        X1_sq = torch.sum(torch.pow(X1, 2), dim=-1) # batch_size x n1
        X2_sq = torch.sum(torch.pow(X2, 2), dim=-1) # batch_size x n2
        X1X2_dot = torch.matmul(X1, X2.transpose(2, 1)) #torch.sum(X1.view(n1, 1, -1) * X2.view(1, n2, -1), dim=-1)

        return X1_sq[:, :, None] + X2_sq[:, None, :] - 2*X1X2_dot


    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1.clone()
        squared_dist = self.compute_distance(X1, X2)
        """
        Computes an n-dimensional Gaussian/RBF kernel matrix by taking points
        in the sample's feature space and maps them to kernel coordinates in
        Hilbert space by calculating the distance to each point in the sample
        space and taking the Gaussian function of the distances.
           K(X,Y) = exp( -(|| X - Y ||^2) / (2 * length_scale^2) )
        where X is the matrix of data points in the sample space,
              Y is the matrix of gaussian centers in the sample space
             sigma is the width of the gaussian function being used
        """
        return torch.exp(-squared_dist/ ( 2 * (self.length_scale**2) ))


class CNNGaussianKernel(nn.Module):
    def __init__(self, input_dim, window_size, embedding_dim, kernel_size,
            n_channels, conv_stride, pool_kernel, length_scale):
        super(CNNGaussianKernel, self).__init__()
        self.feature_extractor = CNN(input_dim, window_size, embedding_dim,
                kernel_size, n_channels, conv_stride, pool_kernel)#feature_extractor
        self.kernel = GaussianKernel(length_scale)

    def forward(self, X1, X2=None):
        batch_size, n_ref, window, steps = X1.shape
        n_test = X2.shape[1] if X2 is not None else None

        X1_enc = self.feature_extractor(X1.view(-1, window,
            steps)).view(batch_size, n_ref, -1)
        X2_enc = self.feature_extractor(X2.view(-1, window,
            steps)).view(batch_size, n_test, -1) if X2 is not None else None

        return self.kernel(X1_enc, X2_enc)
