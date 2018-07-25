import pandas as pd
import csv
import time
import argparse
import numpy as np
from datetime import timedelta, datetime
import pickle as pk
import csv

nan = np.nan
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, help="window length in minute",
        default=10)
args = parser.parse_args()


window = timedelta(minutes=args.window)

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

with open('n1lounge8f_06.csv', 'r') as f:
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

features = list(sorted(features))
columns += features

features = {feature:i for i, feature in enumerate(features)}

print(columns)
input()

data = np.empty((0, len(columns)), dtype=object)


with open('n1lounge8f_06.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row_ in csv_reader:
        key = (row_[0], row_[1])
        value = row_[2]
        value = typecast(value)
        if value is None:
            continue
        timestamp = int(row_[3]) // 1000
        timestamp = datetime.fromtimestamp(timestamp)
        if window_start is None:
            window_start = timestamp
            row['timestamp'] = window_start.timestamp()
            window_end = timestamp + window
        if window_end < timestamp:
            record = {key: averaging_with_timestamp(row[key]) for key in row.keys()}
            data = np.append(data, np.array([[record[key] for key in columns]],
                dtype=object), axis=0)  # record history
            while window_end < timestamp: # jumps
                window_start = window_end
                window_end = window_start + window
                row['timestamp'] = window_start.timestamp()
                # row init
                row = {key: [(window_start.timestamp(), record[key])] if (isinstance(row[key], list) and row[key]) else row[key]
                        for key in row.keys()}
        # take the last value
        if isinstance(row[key], list):
            row[key].append((timestamp.timestamp(), value[0]))
        else:
            row[key] = value


df = pd.DataFrame(data, columns=columns)
df.set_index('timestamp')

describe = df.describe()
#categorical = (describe.loc['unique'] <= 2)
#categorical = [col for col in df.columns if categorical[col]]
#print(categorical)
#for col in categorical:
#    df[col] = df[col].astype('category')
#    df[col] = df[col].cat.codes.replace(-1, nan)


#df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
print(df)
print(data)

with open('n1lounge8f_06_{}.df'.format(args.window), 'wb') as f:
   pk.dump(df, f)

df.to_csv('n1lounge8f_06_{}.csv'.format(args.window))
