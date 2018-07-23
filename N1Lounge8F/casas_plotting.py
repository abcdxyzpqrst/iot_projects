import pandas
import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

if __name__ == "__main__":
    cnt = 0
    dates = []
    with open("iot_projects/data/processed/kyoto_60.csv", 'r') as f:
        for line in f.readlines():
            if cnt == 0:
                cnt += 1
                continue
            else:
                items = line.split(',')
                dates.append(np.datetime64(items[1]))

    years = mdates.YearLocator()
    months = mdates.MonthLocator()
    yearsFmt = mdates.DateFormatter('%Y')

    data = pandas.read_csv('iot_projects/data/processed/kyoto_60.csv')
    cols = data.columns
    
    flag = 1                    # flag == 1,2 means irrelevant column names (not sensor names)
    cnt = 0                     # cnt == k means that you draw 'k' sensor plots in one screen
    fig = plt.figure()
    for name in cols:
        if flag == 1:
            flag += 1
            continue
        elif flag == 2:
            flag += 1
            continue
        cnt += 1
        tmp = np.asarray(data[name])

        ax = plt.subplot(8, 1, cnt)
        ax.plot(dates, tmp)
         
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        
        datemin = np.datetime64(dates[0], 'Y')
        datemax = np.datetime64(dates[-1], 'Y') + np.timedelta64(1, 'Y')
        ax.set_xlim(datemin, datemax)
        ax.set_ylabel(name)
        ax.format_xdata = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.grid(True)

        if cnt == 8:
            fig.autofmt_xdate()
            plt.show()
            cnt = 0
        else:
            pass
    main()
