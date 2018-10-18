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

task2index = dict({})
index = 0
task_index = 0

columns = ['timestamp', 'task']

window_start = None
window_end = None


features = set({})
tasks = set({})

tasks = ['Clean',
   'Meal_Preparation',
   'Bed_to_Toilet',
   'Personal_Hygiene',
   'Sleep',
   'Work',
   'Study',
   'Wash_Bathtub',
   'Watch_TV']

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

with open('raw/casas/twor.2009/data', 'r') as f:
    for line in f:
        line = line.split()
        feature = line[2]
        features.add(feature)

columns += list(sorted(features))
tasks = list(sorted(tasks))
features = list(sorted(features))

tasks = {task:i for i, task in enumerate(tasks)}
features = {feature:i for i, features in enumerate(features)}

data = np.empty((0, len(columns)), dtype=object)
row = {col: nan for col in columns}
row['task'] = (nan, nan)

R1_task2nan = True
R2_task2nan = True
with open('raw/casas/twor.2009/data', 'r') as f:
    for line in f:
        line = line.split()
        line[1] = line[1].split(".")[0]
        #print(line)
        time = line[:2]
        timestamp = datetime.strptime(" ".join(time), "%Y-%m-%d %H:%M:%S")
        #print(timestamp)
        if window_start is None:
            window_start = timestamp
            window_end = timestamp + window
        if window_end < timestamp:
            #print(row)
            data = np.append(data, np.array([[row[key] for key in columns]],
                dtype=object), axis=0)
            #print(data)
            #input()
            if R1_task2nan:
                row['task'] = (nan, row['task'][1])
            if R2_task2nan:
                row['task'] = (row['task'][0], nan)
            while window_end < timestamp: # jumps
                window_start = window_end
                window_end = window_start + window
                row['timestamp'] = window_start.timestamp()
                #data.append(np.array([[row[key] for key in columns]]))
        row[line[2]] = typecast(line[3])
        if len(line) > 4:
            if line[5] == 'begin':
                if 'R1' in line[4]:
                    row['task'] = (tasks[line[4][3:]], row['task'][1])
                    R1_task2nan = False
                elif 'R2' in line[4]:
                    row['task'] = (row['task'][0], tasks[line[4][3:]])
                    R2_task2nan = False
                else:
                    row['task'] = (tasks[line[4]], tasks[line[4]])
                    R1_task2nan = R2_task2nan = False
            else:
                if 'R1' in line[4]:
                    R1_task2nan = True
                elif 'R2' in line[4]:
                    R2_task2nan = True
                else:
                    R1_task2nan = R2_task2nan = True

df = pd.DataFrame(data, columns=columns)
df.set_index('timestamp')

describe = df.describe()
categorical = (describe.loc['unique'] <= 2)

categorical = [col for col in df.columns if categorical[col]]
print(categorical)
for col in categorical:
    df[col] = df[col].astype('category')
    df[col] = df[col].cat.codes.replace(-1, nan)


#df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
print(df)
print(data)

with open('processed/casas_{}.df'.format(args.window), 'wb') as f:
   pk.dump(df, f)

df.to_csv('processed/casas_{}.csv'.format(args.window))
