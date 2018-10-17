import numpy as np

def f1_score(scores, y, thrs, window):
    y_est = np.where((scores > thrs), 1, 0)
    tp = 0
    fn = 0
    fp = 0
    for i, score in enumerate(scores): # tp, fp 계산
        if score > thrs:
            s = max(i - window, 0)
            if np.any(y[s:i] == 0):
                fp += 1

    for i, y_ in enumerate(y):
        if y_ == 1:
            e = min(i + window, len(scores))
            if np.all(scores[i:e] < thrs):
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
    return precision, recall, f1
