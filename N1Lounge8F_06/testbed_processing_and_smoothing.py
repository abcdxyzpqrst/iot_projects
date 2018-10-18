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
parser.add_argument("--window", type=int, help="window length in seconds",
        default=10)
args = parser.parse_args()


#window = timedelta(minutes=args.window)
window = timedelta(seconds=args.window)

def typecast(value):
    try:
        value = float(value)
        return [value]
    except:
        if value == 'true':
            return 1
        if value == 'false':
            return 0
        if value == 'On':
            return 1
        if value == 'Off':
            return 0
        #if value == 'null':
        #    return None
        return None

def row_init(prev_row, timestamp):
    #init_row = {c:prev_row[c] for c in prev_row if c[0] in ["P", "I", "L"]}
    init_row.update({c:0 for c in prev_row if c[0] in ["A", "D", "M"]})
    init_row.update({c:[] for c in prev_row if c[0] in ["T"]})
    init_row['timestamp'] = timestamp
    assert len(prev_row) == len(init_row)
    return init_row


def averaging_with_timestamp(lst):
    if not isinstance(lst, list):
        return lst
    if lst:
        if len(lst) == 1:
            return lst[0][1]
        area = 0
        for i in range(len(lst) - 1):
            t1, v1 = lst[i]
            t2, v2 = lst[i+1]
            h = t2 - t1
            v = v1 + v2
            area += v * h / 2
        assert (lst[-1][0] - lst[0][0]) <= window.total_seconds()
        last_interv = window.total_seconds() - (lst[-1][0] - lst[0][0])
        area += last_interv * lst[-1][1]
        value = float(area) / window.total_seconds()
        return value
    else:
        return nan


index = 0

columns = ['timestamp']

window_start = None
window_end = None


features = set({})

"""
The sensors can be categorized by:

   Mxx:       motion sensor (ON/OFF)
   Lxx: ON / OFF
   Ixx:       item sensor for selected items in the kitchen PRESENT/ABSENT
   Dxx:       door sensor (OPEN/CLOSE)
   AD1-A:     burner sensor (float)
   AD1-B:     hot water sensor (float)
   AD1-C:     cold water sensor (float)
"""

row = {col: nan for col in columns}

all = glob.glob("N1Lounge8F_2018-06-*.csv")
all = sorted(all) # time order

for filename in all:
 with open(filename, 'r') as f:
    csv_reader = csv.reader(f)
    for row_ in csv_reader:
        if row_[0] == 'WebCamAgent':
            continue
        key = (row_[0], row_[1])
        value = row_[2]
        value = typecast(value)
        if value is not None:
            features.add(key)
        if isinstance(value, list):
            row[key] = []
        else:
            row[key] = nan
        timestamp = int(row_[3])
        #print(key, value, timestamp)
row['changepoint'] = 0 # change point (1) or not (0)

features = list(sorted(features))
columns += features
columns += ["changepoint"]
features = {feature:i for i, feature in enumerate(features)}

print(columns)
input()

data = np.empty((0, len(columns)), dtype=object)


for filename in all:
  print(filename)
  window_start = None
  window_end = None
  with open(filename, 'r') as f:
    csv_reader = csv.reader(f)
    row['changepoint'] = 1
    for row_ in csv_reader:
        key = (row_[0], row_[1])
        value = row_[2]
        value = typecast(value)
        if value is None:
            continue
        timestamp = int(row_[3]) // 1000
        timestamp = datetime.fromtimestamp(timestamp)
        if window_start is None: # start of the file
            window_start = timestamp
            row['timestamp'] = window_start.timestamp()
            window_end = timestamp + window
            # row init
            print("row init")
            row = {key: [(window_start.timestamp(), record[key])] if (isinstance(row[key], list) and row[key])
                    else row[key] for key in row.keys()}
        if window_end < timestamp:
            record = {key: averaging_with_timestamp(row[key]) for key in row.keys()}
            data = np.append(data, np.array([[record[key] for key in columns]],
                dtype=object), axis=0)  # record history
            print(record['changepoint'])
            row['changepoint'] = 0
            while window_end < timestamp: # jumps
                window_start = window_end
                window_end = window_start + window
                row['timestamp'] = window_start.timestamp()
                # row init
                row = {key: [(window_start.timestamp(), record[key])] if (isinstance(row[key], list) and row[key]) else row[key]
                        for key in row.keys()}
        if isinstance(row[key], list):
            row[key].append((timestamp.timestamp(), value[0]))
        else:
            # take the last value
            row[key] = value

    record = {key: averaging_with_timestamp(row[key]) for key in row.keys()}
    data = np.append(data, np.array([[record[key] for key in columns]],
        dtype=object), axis=0)  # record history

df = pd.DataFrame(data, columns=columns)
df.set_index('timestamp')

smoothing_list = \
[('AirMonitorAgent', 'CarbonDioxide')]#, #conti
#('AirMonitorAgent', 'ParticulateMatter'), #conti
#('AirMonitorAgent', 'VolatileCompounds'), #conti
#('DoorAgent', 'CorrectedUserCount'), #integer
#('LoungeMonnitServerAgent', 'totalSeatCount'), #integer
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
'''
for key in smoothing_list:
    x = np.arange(len(df['timestamp'].values)) #/ 1000000
    y = df[key].values
    x = x[pd.notnull(y)]
    y = y[pd.notnull(y)]

    x_int = np.linspace(x[0], x[-1], 1000)

    #gp = GaussianProcessRegressor(kernel=None, alpha=1e-10)
    #gp.fit(x.reshape(-1, 1), y)
    #y_int = gp.predict(x_int.reshape(-1, 1))

    tck = interpolate.splrep(x, y, k=3, s=100000)
    y_int = interpolate.splev(x_int, tck, der=0)
    y_smoothing = interpolate.splev(x, tck, der=0)
    df.loc[x, key] = y_smoothing
    #print(df[key].loc[x])
    #print(y_smoothing)
    #input()

    #print(y_int)
    fig = plt.figure(str(key))
    plt.subplot(111)
    plt.plot(x, y, marker='o', linestyle='', alpha=0.5, color='r')
    plt.plot(x_int, y_int, linestyle='-', linewidth=0.75, color='k')
    plt.xlabel("X")
    plt.ylabel("Y")
plt.show()

#df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
print(df)
print(data)
'''
with open('n1lounge8f_06_{}sec.df'.format(args.window), 'wb') as f:
   pk.dump(df, f)

df.to_csv('n1lounge8f_06_{}sec.csv'.format(args.window))



