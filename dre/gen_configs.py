import yaml
from itertools import product
import os
import numpy as np

validation_split = [0.5]
window = [60]
jump = [3]
group_size = [50]
batch_size = [100]
embedding_dim = [30]
conv_kernel = [15]
n_channels = [[50, 30, 10]]
conv_stride = [1]
pool_kernel = [1]
length_scale = np.linspace(0.5, 1.5, num=5)
alpha = [0.01]
reg_lambda = [0.01, 0.1, 1]
n_epochs = [100]


params = [
    'validation_split',
    'window',
    'jump',
    'group_size',
    'batch_size',
    'embedding_dim',
    'conv_kernel',
    'n_channels',
    'conv_stride',
    'pool_kernel',
    'length_scale',
    'alpha',
    'reg_lambda',
    'n_epochs']

for i, conf in enumerate(product(validation_split,
                                 window,
                                 jump,
                                 group_size,
                                 batch_size,
                                 embedding_dim,
                                 conv_kernel,
                                 n_channels,
                                 conv_stride,
                                 pool_kernel,
                                 length_scale,
                                 alpha,
                                 reg_lambda,
                                 n_epochs)):

    config = {key:value for key, value in zip(params, conf)}
    config['idx'] = i
    config['plot'] = False
    print(config)
    if not os.path.exists('configs'):
        os.makedirs('configs')
    with open('configs/{}.yaml'.format(i), 'w') as file:
        yaml.dump(config, file)
