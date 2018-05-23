import pandas as pd
import csv
import time
import argparse
import numpy as np
from datetime import timedelta, datetime
import pickle as pk
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
        return value

def row_init(prev_row, timestamp):
    init_row = {c:prev_row[c] for c in prev_row if c[0] in ["P", "I", "T", "L"]}
    init_row.update({c:0 for c in prev_row if c[0] in ["A", "D", "M"]})
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

with open('raw/casas/twor.2009/raw', 'r') as f:
    for line in f:
        line = line.split()
        feature = line[2]
        if feature[0] not in ["E", "R", "F"]:
            features.add(feature)

features = list(sorted(features))
columns += features

features = {feature:i for i, features in enumerate(features)}

data = np.empty((0, len(columns)), dtype=object)

row = {col: nan for col in columns}

with open('raw/casas/twor.2009/raw', 'r') as f:
    for line in f:
        line = line.split()
        line[1] = line[1].split(".")[0]
        time = line[:2]
        timestamp = datetime.strptime(" ".join(time), "%Y-%m-%d %H:%M:%S")
        if window_start is None:
            window_start = timestamp
            window_end = timestamp + window
            row = row_init(row, window_start)
        if window_end < timestamp:
            data = np.append(data, np.array([[row[key] for key in columns]],
                dtype=object), axis=0)  # record history
            while window_end < timestamp: # jumps
                window_start = window_end
                window_end = window_start + window
                row = row_init(row, window_start)
                #row['timestamp'] = window_start.timestamp()
        try:
            if line[2][0] == 'D' and line[3] == 'OPEN':
                row[line[2]] += 1
            elif line[2][0] == 'M' and line[3] == 'ON':
                row[line[2]] += 1
            elif line[2][0] == 'A':
                row[line[2]] = float(line[3])
            elif line[2][0] == "P":
                row[line[2]] = float(line[3])
            elif line[2][0] == "I":
                row[line[2]] = 1 if line[3] == "PRESENT" else 0
            elif line[2][0] == "T":
                row[line[2]] = float(line[3])
            elif line[2][0] == "L":
                row[line[2]] = 1 if line[3] == "ON" else 0
            else:
                if line[2][0] not in 'DMAPITL':
                    print("IGNORE", line)
        except:
            print(line)

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

with open('processed/casas_raw_{}.df'.format(args.window), 'wb') as f:
   pk.dump(df, f)

df.to_csv('processed/casas_raw_{}.csv'.format(args.window))
