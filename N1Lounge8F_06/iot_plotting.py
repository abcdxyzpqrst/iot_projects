import numpy as np
import pandas
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from datetime import datetime

#cols =\
#[('AirMonitorAgent', 'CarbonDioxide'), #conti
#('AirMonitorAgent', 'ParticulateMatter'), #conti
#('AirMonitorAgent', 'VolatileCompounds'), #conti
#('DoorAgent', 'CorrectedUserCount'), #integer
#('LoungeMonnitServerAgent', 'totalSeatCount'), #integer
#('SoundSensorAgent', 'SoundFridge'), #conti
#('SoundSensorAgent', 'SoundRightWall3'), #conti
#('SoundSensorAgent', 'SoundTV')]#, conti
#
def main():
    data = pandas.read_csv("n1lounge8f_06_nonempty.csv")
    cols = data.columns
    #print(cols)
    #x_datetime = data['timestamp'].values
    #x_datetime = [np.datetime64(datetime.fromtimestamp(x), 's') for x in x_datetime]

    task_label = data['changepoint'].values # pandas.read_csv("Task_201806.csv", header=None)
    changepoint, = np.where(task_label == 1)
    x = np.arange(len(task_label))

    #date = task_label[0].values
    #start_time = task_label[1].values
    #end_time = task_label[2].values
    #task = task_label[3].values
    #n_people = task_label[4].values

    #s_datetime = list(map(lambda x: np.datetime64(x, 's') , [datetime.strptime("{} {}".format(d, st), "%Y-%m-%d %H:%M:%S")
    #    for d, st in zip(date, start_time)]))
    #e_datetime = list(map(lambda x: np.datetime64(x, 's') , [datetime.strptime("{} {}".format(d, et), "%Y-%m-%d %H:%M:%S")
    #    for d, et in zip(date, end_time)]))

    #s_datetime = np.array(s_datetime)
    #e_datetime = np.array(e_datetime)
    #print(s_datetime)
    #print(e_datetime)

    #years = mdates.YearLocator()
    #months = mdates.MonthLocator()
    #hours = mdates.HourLocator()
    #minutes = mdates.MinuteLocator()
    #monthsFmt = mdates.DateFormatter('%Y-%m')
    #hoursFmt = mdates.DateFormatter('%Y-%m-%d %H')

    #timestamp = data['timestamp'].values
    #print(timestamp)
    #input()

    flag = 1
    cnt = 0
    fig = plt.figure()
    for name in cols:
        if name == 'timestamp':
            continue
        if name == 'changepoint':
            continue

        if flag == 1:
            flag += 1
        elif flag == 2:
            flag += 1
            continue
        cnt += 1
        tmp = np.asarray(data[str(name)])

        ax = plt.subplot(8, 1, cnt)
        ax.plot(x, tmp)

        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        #ax.xaxis.set_minor_locator(hours)
        #ax.xaxis.set_minor_formatter(hoursFmt)

        #datemin = x_datetime[0]
        #datemax = x_datetime[-1]
        #ax.set_xlim(datemin, datemax)
        ax.set_ylabel(name)
        #ax.format_xdata = hoursFmt
        ax.set_xticks([])
        ax.set_title(name)
        ax.grid(True)
        for cp in changepoint:
            ax.axvline(cp, alpha=0.3, color='red')
            #ax.axvline(ed, alpha=0.3, color='blue')
        if cnt == 8:
            plt.show()
            cnt = 0
        else:
            pass
    plt.show()
    return

if __name__ == "__main__":
    main()
