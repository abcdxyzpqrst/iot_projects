import os, sys
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

data = pd.read_csv('data', sep=' ', header=None, names=["date", "time", "sensor id", "sensor value", "task", "begin/end"])

sensor_ids = list(data['sensor id'].unique())
sensor_ids.remove('LEAVEHOME')
sensor_ids.remove('ENTERHOME')
# ['M003', 'T002', 'T003', 'T004', 'T005', 'T001',
#  'M002', 'M007', 'M005', 'M004', 'M006', 'M008',
#  'M020', 'M010', 'M011', 'M012', 'M013', 'M014',
#  'M009', 'M018', 'M019', 'M015', 'M016', 'M017',
#  'M021', 'M022', 'M023', 'M001', 'M024', 'D002',
#  'M031', 'D004', 'M030', 'M029', 'M028', 'D001', 'M026', 'M027', 'M025']
sensor2idx = {sensor:i for i, sensor in enumerate(sensor_ids)}
sensor2count = {sensor:0 for sensor in sensor_ids} # should be initialize every "freq" times
prev_sensor2count = {sensor:0 for sensor in sensor_ids}
tasks = data['task'].dropna().unique()

print(sensor_ids)
print(tasks)

window = 30
count = 0

data_field = sensor_ids + [sensor + "_count" for sensor in sensor_ids] + \
        [sensor + "_elapsed" for sensor in sensor_ids] +\
        ['day_of_week', 'hour_of_day', 'sec_past_midn'] + \
        ['first_sensor', 'last_sensor', 'duration', 'freq_sensor',
        'elapsed_from_last'] + ['change_point']

processed = pd.DataFrame(columns=data_field)

# window features
# 1. most recent sensor in window,
# 2. first sensor in window,
# 3. window duration,
# 4. most frequent sensors from previous two windows,
# #5. last sensor location in window,
# #6. last motion sensor location in window,
# 7. entropy-based data complexity of window,
# 8. time elapsed since last sensor event in window
# =================================================
# timing feature
# 1. day of week,
# 2. hour of day,
# 3. seconds past midnight
# ================================================
# raw
# infrared motion (ON/OFF), magnetic door
# (OPEN/CLOSE), temperature (continuous)

window_dict = {field: None for field in data_field}
for sensor_ in sensor_ids:
    if sensor_[0] in ['M', 'D']:
        window_dict[sensor_] = 0
    else: # Temperature
        window_dict[sensor_] = None
window_dict['change_point'] = 0


date_and_time = None
for index, row in tqdm(data.iterrows()):
    try:
        date_and_time = datetime.strptime("{} {}".format(row['date'], row['time']),
                '%Y-%m-%d %H:%M:%S.%f')
    except:
        input("exception")
        date_and_time = datetime.strptime("{} {}".format(row['date'], row['time']),
                '%Y-%m-%d %H:%M:%S')
    if index == 0:
        sensor2time = {sensor:date_and_time for sensor in sensor_ids}
    #time = datetime.strptime(row['time'], '%H:%M:%S.%f')
    #print(date, time)
    day_of_week = date_and_time.weekday()
    hour_of_day = date_and_time.hour
    midnight = date_and_time.replace(hour=0, minute=0, second=0, microsecond=0)
    sec_past_midn = (date_and_time - midnight).seconds
    sensor = row['sensor id']
    val = row['sensor value']
    task = row['task']
    if sensor not in sensor_ids:
        continue

    if count == 0:
        if index > 0:
            # append data here
            for sensor_ in sensor_ids:
                window_dict[sensor_ + '_count'] = sensor2count[sensor_]
                window_dict[sensor_ + '_elapsed'] = (date_and_time - sensor2time[sensor_]).seconds
            max_freq = 0
            for sensor_ in sensor_ids:
                freq = sensor2count[sensor_] + prev_sensor2count[sensor_]
                if max_freq < freq:
                    max_freq = freq
                    window_dict['freq_sensor'] = sensor2idx[sensor_]

            prev_sensor2count = sensor2count
            sensor2count = {s:0 for s in sensor2count}
            window_dict['elapsed_from_last'] = (date_and_time - prev_date_and_time).seconds
            window_dict['duration'] = (date_and_time - window_start).seconds

            processed = processed.append(window_dict, ignore_index=True)
            #print("ADD", window_dict)
            processed.to_csv("processed.csv")
            #print(processed)
            for field in window_dict:
                if field not in sensor_ids + ['change_point']:
                    window_dict[field] = None
                window_dict['change_point'] = 0
                    #if field[0] in ['M', 'D']:
                    #    window_dict[field] = 0
                #else:
                #    window_dict[field] = None  # init
        #window_dict = {field:None for field in window_dict} # init
        window_dict['first_sensor'] = sensor2idx[sensor]
        window_dict['day_of_week'] = day_of_week
        window_dict['hour_of_day'] = hour_of_day
        window_dict['sec_past_midn'] = sec_past_midn
        window_start = date_and_time
    count += 1

    if sensor[0] == "T":
        window_dict[sensor] = val
    if sensor[0] == "M":
        window_dict[sensor] = 1 if val  == 'ON' else 0
    if sensor[0] == "D":
        window_dict[sensor] = 1 if val == 'OPEN' else 0
    if task in tasks:
        window_dict['change_point'] = 1

    sensor2count[sensor] = 1 + sensor2count[sensor]
    sensor2time[sensor] = date_and_time

    if count % window == 0:
        window_dict['last_sensor'] = sensor2idx[sensor]
        count = 0
        #sensor2count = {sensor: 0 for sensor in sensor2count}
    prev_date_and_time = date_and_time


