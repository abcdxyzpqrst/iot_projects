import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from collections import OrderedDict


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):
    def __init__(self, input_dim, window_size, embedding_dim, kernel_size,
            n_channels, conv_stride, pool_kernel):
        super(CNN, self).__init__()

        #self.MODEL = kwargs["MODEL"]
        self.input_dim = input_dim # kwargs["input_dim"] # d
        self.window_size = window_size # kwargs["window_size"] # k
        self.embedding_dim = embedding_dim # kwargs["embedding_dim"]
        self.kernel_size = kernel_size # kwargs["kernel_size"] # default
        self.n_channels = n_channels #  kwargs["n_channels"]
        self.conv_stride = conv_stride # kwargs["conv_stride"]
        self.pool_kernel = pool_kernel # kwargs["pool_kernel"]
        m = OrderedDict()
        def update_L(L, kernel_size, dilation=1, padding=0, stride=2):
            # calc kernel
            assert L >= kernel_size, " kernel_size {} should be smaller than L {}".format(kernel_size, L)
            new_L = int((L + 2 * padding - dilation * (kernel_size - 1) - 1) /
                    stride + 1)
            return new_L

        ### 1
        L = self.window_size
        m['conv1'] = nn.Conv1d(self.input_dim, self.n_channels[0],
                self.kernel_size, stride=self.conv_stride)
        L = update_L(L, self.kernel_size, stride=self.conv_stride)
        m['relu1'] = nn.ReLU()
        if self.pool_kernel > 1:
            m['maxpool1'] = nn.MaxPool1d(kernel_size=self.pool_kernel)
            L = update_L(L, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        ### 2
        m['conv2'] = nn.Conv1d(self.n_channels[0], self.n_channels[1],
                self.kernel_size, stride=self.conv_stride)
        L = update_L(L, self.kernel_size, stride=self.conv_stride)
        m['relu2'] = nn.ReLU()
        if self.pool_kernel > 1:
            m['maxpool2'] = nn.MaxPool1d(kernel_size=self.pool_kernel)
            L = update_L(L, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        ### 3
        m['conv3'] = nn.Conv1d(self.n_channels[1], self.n_channels[2],
                self.kernel_size, stride=self.conv_stride)
        L = update_L(L, self.kernel_size, stride=self.conv_stride)
        m['relu3'] = nn.ReLU()
        if self.pool_kernel > 1:
            m['maxpool3'] = nn.MaxPool1d(kernel_size=self.pool_kernel)
            L = update_L(L, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        self.feature_extraction = nn.Sequential(m)

        print("Final L", L)
        fc_in = self.n_channels[-1] * L
        self.fc = nn.Linear(fc_in, self.embedding_dim)

        #self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]

    def forward(self, x):
        #print(x.type())
        #input()
        x = x.transpose(1, 2)
        x = self.feature_extraction(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    '''
        # one for UNK and one for zero padding
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')
    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x
    '''
