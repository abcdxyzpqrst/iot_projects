import numpy as np
import pandas
from matplotlib import pyplot as plt

def main():
    data = pandas.read_csv("n1lounge8f_06_10.csv")
    cols = data.columns
    
    flag = 1
    cnt = 0
    fig = plt.figure()
    for name in cols:
        if flag == 1:
            flag += 1
        elif flag == 2:
            flag += 1
            continue
        cnt += 1
        tmp = np.asarray(data[name])

        ax = plt.subplot(8, 1, cnt)
        ax.plot(np.arange(len(tmp)), tmp)
        ax.set_xticks([])
        ax.set_title(name)
        ax.grid(True)

        if cnt == 8:
            plt.show()
            cnt = 0
        else:
            pass
    return

if __name__ == "__main__":
    main()
