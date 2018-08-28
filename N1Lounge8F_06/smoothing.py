import pandas as pd
import csv
import time
import argparse
import numpy as np
from datetime import timedelta, datetime
import pickle as pk
import csv
from scipy import interpolate
import matplotlib.pyplot as plt
import os
import glob

nan = np.nan
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, help="window length in minute",
        default=1)
args = parser.parse_args()

with open('n1lounge8f_06_{}.df'.format(args.window), 'rb') as f:
   df = pk.load(f)

#[('AirMonitorAgent', 'CarbonDioxide')]#, #conti
#[('AirMonitorAgent', 'ParticulateMatter'), 1000]#, #conti 1000 or 10000
#[('AirMonitorAgent', 'VolatileCompounds'), 1e6]#, #conti 1e6
smoothing_list = \
[('DoorAgent', 'CorrectedUserCount')]#, 10], #integer
#[('LoungeMonnitServerAgent', 'totalSeatCount')] #integer
#('SoundSensorAgent', 'SoundFridge'), #conti
#('SoundSensorAgent', 'SoundRightWall3'), #conti
#('SoundSensorAgent', 'SoundTV')] #conti

'''
[('LoungeMonnitServerAgent', 'seat1'), #binary
('LoungeMonnitServerAgent', 'seat2'), #binary
('LoungeMonnitServerAgent', 'seat3'), #binary
('LoungeMonnitServerAgent', 'seat4'), #binary
('LoungeMonnitServerAgent', 'seat5a'), #binary
('LoungeMonnitServerAgent', 'seat5b'), #binary
('LoungeMonnitServerAgent', 'seat5c'), #binary
('LoungeMonnitServerAgent', 'seat6a'), #binary
('LoungeMonnitServerAgent', 'seat6b'), #binary
('LoungeMonnitServerAgent', 'seat6c'), #binary
('SoundSensorAgent', 'SoundFridge'),
('SoundSensorAgent', 'SoundRightWall3'),
('SoundSensorAgent', 'SoundTV'),
('LoungeMonnitServerAgent', 'MDoor'), #binary
('LoungeMonnitServerAgent', 'MLight'), #binary
('LoungeMonnitServerAgent', 'MRunningMachine'), #binary
('LoungeMonnitServerAgent', 'MSofa'), #binary
('LoungeMonnitServerAgent', 'MStair'), #binary
('LoungeMonnitServerAgent', 'MVendingMachine'), #binary
('DoorAgent', 'UserCount'), #integer
('DoorAgent', 'CorrectedUserCount')] #integer
'''

M_list = \
[('LoungeMonnitServerAgent', 'MDoor'), #binary
('LoungeMonnitServerAgent', 'MLight'), #binary
('LoungeMonnitServerAgent', 'MRunningMachine'), #binary
('LoungeMonnitServerAgent', 'MSofa'), #binary
('LoungeMonnitServerAgent', 'MStair'), #binary
('LoungeMonnitServerAgent', 'MVendingMachine')] #binary

from sklearn.gaussian_process import GaussianProcessRegressor

df_ = pd.rolling_mean(df, 30)
for key in smoothing_list:
    x = np.arange(len(df['timestamp'].values)) #/ 1000000
    y = df[key].values.reshape(-1)
    y_ = df_[key].values.reshape(-1)
    print(y)
    x = x[pd.notnull(y)]
    y_ = y_[pd.notnull(y)]
    y = y[pd.notnull(y)]

    x_int = np.linspace(x[0], x[-1], 1000)

    #gp = GaussianProcessRegressor(kernel=None, alpha=1e-10)
    #gp.fit(x.reshape(-1, 1), y)
    #y_int = gp.predict(x_int.reshape(-1, 1))

    tck = interpolate.splrep(x, y, k=3, s=10)
    y_int = interpolate.splev(x_int, tck, der=0)
    #y_smoothing = interpolate.splev(x, tck, der=0)
    df.loc[x, key] = y_#.reshape(-1, 1)
    #print(df[key].loc[x])
    #print(y_smoothing)
    #input()

    #print(y_int)
    fig = plt.figure(str(key))
    plt.subplot(111)
    plt.plot(x, y, marker='o', linestyle='', alpha=0.5, color='r')
    plt.plot(x, y_, linestyle='-', color='b')
    #plt.plot(x_int, y_int, linestyle='-', linewidth=0.75, color='k')
    plt.xlabel("X")
    plt.ylabel("Y")
plt.show()

#df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
print(df)

with open('n1lounge8f_06_{}_smoothing.df'.format(args.window), 'wb') as f:
   pk.dump(df, f)

df.to_csv('n1lounge8f_06_{}_smoothing.csv'.format(args.window))



