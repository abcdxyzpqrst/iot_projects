import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from datetime import datetime

def main():
    data = pandas.read_csv("../../N1Lounge8F/n1lounge8f_06_1.csv")
    cols = data.columns
    
    x_datetime = data['timestamp'].values[30:10037]
    x_datetime = [np.datetime64(datetime.fromtimestamp(x), 's') for x in x_datetime]
    
    y = np.load("score.npy")
    print (x_datetime[2332])
    task_label = pandas.read_csv("../../N1Lounge8F/Task_201806.csv", header=None)

    date = task_label[0].values
    start_time = task_label[1].values
    end_time = task_label[2].values
    task = task_label[3].values
    n_people = task_label[4].values

    s_datetime = list(map(lambda x: np.datetime64(x, 's') , [datetime.strptime("{} {}".format(d, st), "%Y-%m-%d %H:%M:%S")
        for d, st in zip(date, start_time)]))
    e_datetime = list(map(lambda x: np.datetime64(x, 's') , [datetime.strptime("{} {}".format(d, et), "%Y-%m-%d %H:%M:%S")
        for d, et in zip(date, end_time)]))

    s_datetime = np.array(s_datetime)
    e_datetime = np.array(e_datetime)
    #print(s_datetime)
    #print(e_datetime)

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    hours = mdates.HourLocator()
    minutes = mdates.MinuteLocator()
    monthsFmt = mdates.DateFormatter('%Y-%m')
    hoursFmt = mdates.DateFormatter('%Y-%m-%d %H')

    timestamp = data['timestamp'].values
    #print(timestamp)
    #input()
    
    fig = plt.figure()
    plt.rc("text", usetex=True)
    plt.rc("font", family="Times New Roman", size=16)

    name1 = "('SoundSensorAgent', 'SoundFridge')"
    name2 = "('SoundSensorAgent', 'SoundTV')"
    name3 = "('SoundSensorAgent', 'SoundRightWall3')"
    
    fig = plt.figure()
    ax1 = plt.subplot(2,1,1)
    
    ax1.plot(x_datetime, np.asarray(data[name1][30:10037]), label="Fridge")
    ax1.plot(x_datetime, np.asarray(data[name2][30:10037]), label="TV")
    ax1.plot(x_datetime, np.asarray(data[name3][30:10037]), label="RightWall3")
    
    ax1.xaxis.set_major_locator(months)
    ax1.xaxis.set_major_formatter(monthsFmt)
    #ax.xaxis.set_minor_locator(hours)
    #ax.xaxis.set_minor_formatter(hoursFmt)

    datemin = x_datetime[0]
    datemax = x_datetime[-1]
    ax1.set_xlim(datemin, datemax)
    #ax1.set_ylabel(name)
    ax1.format_xdata = hoursFmt
    ax1.set_xticks([])
    ax1.set_title("Sound Sensor Signals")
    ax1.grid(True)
    for sd, ed in zip(s_datetime, e_datetime):
        ax1.axvline(sd, alpha=0.3, color='red')
        ax1.axvline(ed, alpha=0.3, color='blue')
    ax1.legend(loc="best")

    ax2 = plt.subplot(2,1,2)
    ax2.plot(x_datetime, y, color="red")
    ax2.plot(x_datetime[3970], 80, marker='x', color="black", markersize=10)
    ax2.set_xlim(datemin, datemax)
    ax2.format_xdata = hoursFmt
    ax2.set_title("Change-Point Score")
    ax2.grid(True)
    plt.show()

    return

if __name__ == "__main__":
    main()
