import numpy as np

def f1_score(scores, y, thrs, window=10):
    y_est = np.where((scores > thrs), 1, 0)
    tp = 0
    fn = 0
    fp = 0

    alarms = []
    detected = False
    cnt = 0
    for i, score in enumerate(scores): # tp, fp 계산
        alarm = 0
        if score <= thrs:
            detected = False
            alarm = 0
            cnt = 0
        elif detected:
            #cnt += 1
            #if cnt % window == 0:
            #    detected = False
            alarm = 0
        #    alarms.append(alarm)
            #continue
        elif score > thrs:
            alarm = 1
            detected = True
            s = max(i - window, 0)
            if np.all(y[s:i] == 0):
                fp += 1
        alarms.append(alarm)

    alarms = np.array(alarms)

    for i, y_ in enumerate(y):
        if y_ == 1:
            e = min(i + window, len(scores))
            if np.all(alarms[i:e] == 0):
                fn += 1
            else:
                tp += 1
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    return precision, recall, f1, alarms
