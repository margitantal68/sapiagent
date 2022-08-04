import os
import pandas as pd
from enum import Enum


class ActionType(Enum):
    # Point Click
    PC = 0
    # Drag Drop
    DD = 1


class SessionType(Enum):
    MIN1 = '1min'
    MIN3 = '3min'


class FeatureType(Enum):
    DX_DY = 'dx_dy'
    DX_DY_DT = 'dx_dy_dt'


STATISTICS = True
c_dataset_path = 'sapimouse'
NUM_MIN_EVENTS = 10
MAX_LEN = 128

# segment the file to meaningful actions
def file2actions(filename, userid):
    df = pd.read_csv(filename)
    df_result = df.loc[df['state'] == 'Released']
    break_points = df_result.index.tolist()
    start_index = 0
    for index in break_points:
        action = df[start_index: index]
        if action.shape[0] < NUM_MIN_EVENTS:
            start_index = index + 1
            continue
        x = action['x']
        y = action['y']
        t = action['client timestamp']
        # button = action['button']
        state = action['state']
        # action_type = ActionType.PC
        if state.tolist()[-1] == 'Drag':
            # remove leading Move events
            action = action.loc[action['state'] == 'Drag']
            if action.shape[0] < NUM_MIN_EVENTS:
                start_index = index + 1
                continue
            x = action['x']
            y = action['y']
            # action_type = ActionType.DD
        rows, cols = action.shape
        if STATISTICS:
            f_action_length.write(str(rows) + '\n')
        indexes = action.index.values
        if STATISTICS:
            time = t[indexes[-1]] - t[indexes[0]] 
            f_action_start_stop.write(str(x[indexes[0]]) + ',' + str(y[indexes[0]]) + ',' + str(x[indexes[-1]]) + ',' + str(y[indexes[-1]]) + ',' + str(rows) + ',' + str(time) + ',' + str(userid) + '\n' )
        start_index = index + 1
        # f_action_start_stop.write()
        # action related features
        features = action2rawfeatures(action)
        features = [str(element) for element in features]
        # features.append( str(action_type.value) )
        features.append(str(userid))
        f_out.write(",".join(features) + '\n')


# input: action - dataframe (client timestamp, button, state, x, y) containing an action
# calculates dx, dy, dt
# fills with zeros until the fixed length (MAX_LEN = 128)
# returns (dx, dy, dt) as a list
def action2rawfeatures(action):
    action = action.drop(columns=['button', 'state'])
    # difference + abs()
    df = action.diff()
    first_row = df.index[0]
    df = df.drop(first_row)
    # df = df.abs()

    dx = df['x']
    dy = df['y']
    dt = df['client timestamp']

    dx = dx.values.tolist()
    dy = dy.values.tolist()
    dt = dt.values.tolist()
    if len(dx) < MAX_LEN:
        # shorter actions: zero padding
        for i in range(MAX_LEN - len(dx)):
            dx.append(0)
            dy.append(0)
            dt.append(0)
    else:
        # longer actions: truncated
        dx = dx[0:MAX_LEN]   
        dy = dy[0:MAX_LEN]   
        dt = dt[0:MAX_LEN]   
    features = [] 
    features.extend(dx)
    features.extend(dy)
    if feature_type == FeatureType.DX_DY_DT:
        features.extend(dt)
    # print(len(features))
    return features


def process_files(session_type):   
    for folder in os.listdir(c_dataset_path):
        # print(folder)
        userid = folder[4:]
        # print(userid)
        folderpath = os.path.join(c_dataset_path, folder)
        # print(folderpath)
        for session_name in os.listdir(folderpath):
            if session_name[-8:-4] == session_type.value:
                sessionpath = os.path.join(folderpath, session_name)
                print('\t' + sessionpath)
                file2actions(sessionpath, userid)


if __name__ == "__main__":
    # BEGIN
    directory = "sapimouse_actions"
    path = os.path.join(".", directory)
    mode = 0o666
    try:
        os.mkdir(path, mode)
    except:
        print(directory + " already exists")
    feature_type = FeatureType.DX_DY
    # END 
    for session_type in SessionType:
        f_out = open('sapimouse_actions/actions_' + session_type.value + '_' + feature_type.value + '.csv', 'w')
        if STATISTICS:
            f_action_length = open('statistics/actions_' + session_type.value + '.csv', 'w')
            f_action_start_stop = open('statistics/actions_start_stop_' + session_type.value + '.csv', 'w')
            f_action_start_stop.write('startx,starty,stopx,stopy,length,time,userid\n')
        process_files(session_type)
        if STATISTICS:
            f_action_start_stop.close()
            f_action_length.close()
        f_out.close()

