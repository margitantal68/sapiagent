import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


# returns histogram of each row
def calculate_histo(df):
    df = df.iloc[:, :-1]
    bins = [0, 10, 20, 30, 40, 100]
    rows, columns = df.shape
    histos = list()
    # print(df.shape)
    for i in range(0, rows):
        df_row = df.iloc[i, :]
        df_row_dx = df_row.iloc[0:128]
        df_row_dy = df_row.iloc[128:256]

        df_row_dx = df_row_dx.abs()
        df_row_dy = df_row_dy.abs()

        count_dx, _ = np.histogram(df_row_dx, bins=bins)
        count_dy, _ = np.histogram(df_row_dy, bins=bins)

        count = np.array(count_dx / 128)
        count = np.append(count, count_dy / 128)
        histos.append(count)
    columns = ["label1" + str(i) for i in range(0, len(count))]
    # print(columns)
    result_df = pd.DataFrame(data=np.vstack(histos), columns=columns)
    return result_df


# returns histogram of each row
def calculate_statistical_features(df):
    df = df.iloc[:, :-1]

    rows, columns = df.shape
    features = []
    # print(df.shape)
    for i in range(0, rows):
        df_row = df.iloc[i, :]
        df_row_dx = df_row.iloc[0:128]
        df_row_dy = df_row.iloc[128:256]

        mean_dx = df_row_dx.mean()
        mean_dy = df_row_dy.mean()

        std_dx = df_row_dx.std()
        std_dy = df_row_dy.std()

        min_dx = df_row_dx.min()
        max_dx = df_row_dx.max()

        min_dy = df_row_dy.min()
        max_dy = df_row_dy.max()

        count = []

        count = np.append(count, min_dx)
        count = np.append(count, max_dx)
        count = np.append(count, mean_dx)
        count = np.append(count, std_dx)

        count = np.append(count, min_dy)
        count = np.append(count, max_dy)
        count = np.append(count, mean_dy)
        count = np.append(count, std_dy)

        features.append(count)
    columns = ["label_histogram" + str(i) for i in range(0, len(count))]
    # print(columns)
    result_df = pd.DataFrame(data=np.vstack(features), columns=columns)
    return result_df


def smoothness(df):
    rows, _ = df.shape
    x = df.iloc[:, 0:128]
    y = df.iloc[:, 128:256]
    x_array = x.values
    y_array = y.values
    features = []
    for i in range(0, rows):
        dx = np.diff(x_array[i, :], axis=0)
        dx = np.abs(dx)
        nx = np.count_nonzero(dx)
        sx = np.sum(dx)

        dy = np.diff(y_array[i, :], axis=0)
        dy = np.abs(dy)
        ny = np.count_nonzero(dy)
        sy = np.sum(dy)

        if nx == 0:
            nx = 1
        if ny == 0:
            ny = 1
        count = []
        count = np.append(count, sx / nx)
        count = np.append(count, sy / ny)
        features.append(count)

    array = np.array(features)
    # print(array)
    m = array.mean(axis=0)
    s = array.std(axis=0)
    # print(m)
    # print(s)
    # print("smoothness_dx: {0:.2f} ({1:.2f})".format(m[0], s[0]))
    # print("smoothness_dy: {0:.2f} ({1:.2f})".format(m[1], s[1]))
    columns = ["label_smoothness" + str(i) for i in range(0, len(count))]
    result_df = pd.DataFrame(data=np.vstack(features), columns=columns)
    return result_df


def movement_efficiency(df):
    rows, _ = df.shape
    x = df.iloc[:, 0:128]
    y = df.iloc[:, 128:256]
    x_array = x.values
    y_array = y.values

    efficiency = []
    for i in range(0, rows):
        # print(x_array[i,:])
        x_coord = [0]
        y_coord = [0]
        for j in range(0, 128):
            x_coord.append(x_coord[-1] + x_array[i, j])
            y_coord.append(y_coord[-1] + y_array[i, j])
        # print(x_coord)
        # print(y_coord)
        distance = dist(x_coord[0], y_coord[0], x_coord[-1], y_coord[-1])
        path = 0
        for k in range(1, 128):
            path += dist(
                x_coord[k], y_coord[k], x_coord[k - 1], y_coord[k - 1]
            )
        try:
            efficiency.append(distance / path)
            # print(str(distance) + ', ' + str(path) + ', ' + str(distance / path))
        except:
            # print(str(distance) + ', ' + str(path))
            # print(x_array[i, :])
            efficiency.append(0.1)
    eff_array = np.array(efficiency)
    m = eff_array.mean()
    s = eff_array.std()
    # print("movement_efficiency: {0:.2f} ({1:.2f})".format(m, s))
    result_df = pd.DataFrame({"label_efficiency": eff_array})
    return result_df


def calculate_features(df):
    # print("*****")
    df1 = movement_efficiency(df)
    # print(df1.shape)
    df2 = smoothness(df)
    # print(df2.shape)
    df3 = calculate_statistical_features(df)
    # print(df3.shape)
    df4 = calculate_histo(df)
    # print(df4.shape)
    df = pd.concat(
        [
            df1.reset_index(drop=True),
            df2.reset_index(drop=True),
            df3.reset_index(drop=True),
            df4.reset_index(drop=True),
        ],
        ignore_index=True,
        axis=1,
    )
    return df


# if __name__ == "__main__":
    # df_train = pd.read_csv("output_actions/actions_1min_dx_dy_dt.csv", header=None)
    # df_train=pd.read_csv("bezier_actions/bezier_actions_1min.csv", header=None)
    # df_train = pd.read_csv("generated_actions/generated_fcn_dx_dy_mse_supervised.csv", header=None)
    # df_train = pd.read_csv(
    #     "generated_actions/generated_bidirectional_dx_dy_mse_supervised.csv",
    #     header=None,
    # )

    # smoothness(df_train)
    # movement_efficiency(df_train)
    # df1 = movement_efficiency(df_train)
    # print(df1)
    # df2 = calculate_histo(df_train)
    # print(df2)
    # df = pd.concat([df1.reset_index(drop=True), df2.reset_index(drop=True)], ignore_index=True, axis=1)
    # print(df)
    # print(df.describe())
