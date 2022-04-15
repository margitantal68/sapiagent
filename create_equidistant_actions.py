import os
import pandas as pd
import numpy as np
from typing import List

OUTPUT_DIR = 'equidistant_actions'
LENGTH = 128


def create_actions(SESSION, dt=False):
    df_start = pd.read_csv('statistics/actions_start_stop_' + SESSION + '.csv')
    f_out = open(OUTPUT_DIR + '/equidistant_' + SESSION + '.csv', 'w')

    startx = df_start['startx']
    starty = df_start['starty']
    stopx = df_start['stopx']
    stopy = df_start['stopy']
    length = df_start['length']
    if dt:
        time = df_start['time']
    
    userid = df_start['userid']
    rows, cols = df_start.shape
    for i in range(rows):
        x1 = startx[i]
        y1 = starty[i]
        x2 = stopx[i]
        y2 = stopy[i]
        n = min(length[i], 128)
        dx = (x2 - x1) / n
        dy = (y2 - y1) / n
        # dxdydt
        if dt:
            dt = (int)(time[i] / n)
     
        # x, y coordinates - floats
        x = []
        y = []
        for i in range(0, n + 1):
            x.append(x1 + i * dx)
            y.append(y1 + i * dy)
        # x, y coordinates - integers
        x_int = [int(a) for a in x]
        y_int = [int(a) for a in y]

        dx_list = np.diff(x_int).tolist()
        dy_list = np.diff(y_int).tolist()

        # dxdydt
        if dt:
            dt_list: List[int] = [dt] * n
     
        dx_list.extend([0] * (128 - n))
        dy_list.extend([0] * (128 - n))
     
        # dxdydt
        if dt:
            dt_list.extend([0] * (128 - n))
        d_list: List[int] = dx_list
        d_list.extend(dy_list)
        # dxdydt
        if dt:
            d_list.extend(dt_list)
        d_list.append(userid[i])
        d_list = [str(e) for e in d_list]
        # if len(d_list) != 257:
        # dxdydt
        # if len(d_list) != 385:
        #     print("{} {}".format(i, len(d_list)))
        f_out.write(",".join(d_list) + ' \n')


if __name__ == "__main__":   
    try:
        # os.mkdir('baseline_actions')
        os.mkdir(OUTPUT_DIR)
    except OSError:
        print('Directory %s already exists' % OUTPUT_DIR)
    else:
        print('Successfuly created the directory %s' % OUTPUT_DIR)
    SESSION = '3min'
    create_actions(SESSION)
    SESSION = '1min'
    create_actions(SESSION)