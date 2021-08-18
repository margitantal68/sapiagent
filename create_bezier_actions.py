import os
import pandas as pd
from pyclick import HumanCurve


MAX_LEN = 128


def generate_trajectories(output_filename, type):
    f_out = open(output_filename, 'w')
    df = pd.read_csv('statistics/actions_start_stop_1min.csv')
    x_start = df['startx']
    y_start = df['starty']
    x_stop = df['stopx']
    y_stop = df['stopy']
    numpoints = df['length']
    userid = df['userid']
    index = x_start.index
    for i in index:
        fromPoint = (x_start[i], y_start[i])
        toPoint = (x_stop[i], y_stop[i])
        user = userid[i]
        print('From: ' + str(fromPoint) + ' To: ' + str(toPoint) + ' numPoints: ' + str(numpoints[i]))
        hc = HumanCurve(fromPoint, toPoint)
        if type == 'baseline':
            points = hc.generateCurve(offsetBoundaryX=10, offsetBoundaryY=50,\
                        leftBoundary=10, rightBoundary=1600, \
                        downBoundary=10, upBoundary=1200, \
                        # knotsCount=0, \
                        # distortionMean=5, distortionStdev=3, distortionFrequency=0.5, \
                        # tween=pytweening.easeOutCubic, \
                        targetPoints= int(numpoints[i]))
        else:
            points = hc.generateCurve(offsetBoundaryX=10, offsetBoundaryY=50,\
                        leftBoundary=10, rightBoundary=1600, \
                        downBoundary=10, upBoundary=1200, \
                        knotsCount=0, \
                        distortionMean=5, distortionStdev=3, distortionFrequency=0.5, \
                        tween=pytweening.easeOutCubic, \
                        targetPoints= int(numpoints[i]))
        points_x = [int(pair[0]) for pair in points]
        points_y = [int(pair[1]) for pair in points]
    
        dx = pd.Series(points_x).diff()
        dy = pd.Series(points_y).diff()

        first_row = dx.index[0]
        dx = dx.drop(first_row)
        dy = dy.drop(first_row)
        
        # Serise --> list
        dx = dx.values.tolist()
        dy = dy.values.tolist()
        if len(dx) < MAX_LEN:
            # shorter are padded with 0s
            for i in range(MAX_LEN - len(dx)):
                dx.append(0)
                dy.append(0)
        else:
            # longer are shortened to MAX_LEN
            dx = dx[0:MAX_LEN]
            dy = dy[0:MAX_LEN]
        features = []
        features.extend(dx)
        features.extend(dy)
        # features.append( str(userid) )
        features = [str(element) for element in features]
        features.append(str(user))
        f_out.write(",".join(features)+'\n')
    f_out.close()


OUTPUT_FOLDER = 'bezier_actions'


if __name__ == "__main__":
    try:
        os.mkdir(OUTPUT_FOLDER)
    except OSError:
        print(OUTPUT_FOLDER + ' folder already exists')
    else:
        print(OUTPUT_FOLDER + ' folder has been created')
    TYPE = 'baseline'
    print('Generate ' + TYPE + ' actions')
    output_filename = OUTPUT_FOLDER + '/' + 'bezier_' + TYPE + '_actions.csv'
    generate_trajectories(output_filename, TYPE)

    TYPE = 'humanlike'
    print('Generate ' + TYPE + ' actions')
    output_filename = OUTPUT_FOLDER + '/' + 'bezier_' + TYPE + '_actions.csv'
    generate_trajectories(output_filename, TYPE)
