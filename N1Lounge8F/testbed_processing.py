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
        return value
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

with open('n1lounge8f_06.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        if row[0] == 'WebCamAgent':
            continue
        key = (row[0], row[1])
        value = row[2]
        value = typecast(value)
        if value is not None:
            features.add(key)
        timestamp = int(row[3])
        #print(key, value, timestamp)

features = list(sorted(features))
columns += features

features = {feature:i for i, feature in enumerate(features)}

print(columns)
input()

data = np.empty((0, len(columns)), dtype=object)

row = {col: nan for col in columns}
with open('n1lounge8f_06.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row_ in csv_reader:
        #if row[0] == 'WebCamAgent':
        #    continue
        key = (row_[0], row_[1])
        value = row_[2]
        value = typecast(value)
        if value is None:
            continue
        timestamp = int(row_[3]) // 1000
        timestamp = datetime.fromtimestamp(timestamp)
        #print(key, value, timestamp)
        #time = line[:2]
        #timestamp = datetime.strptime(" ".join(time), "%Y-%m-%d %H:%M:%S")
        if window_start is None:
            window_start = timestamp
            window_end = timestamp + window
            #row = row_init(row, window_start)
        if window_end < timestamp:
            data = np.append(data, np.array([[row[key] for key in columns]],
                dtype=object), axis=0)  # record history
            while window_end < timestamp: # jumps
                window_start = window_end
                window_end = window_start + window
                #row = row_init(row, window_start)
                row['timestamp'] = window_start.timestamp()
        # take the last value
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
