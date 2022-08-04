import settings as stt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper", font_scale=2)
    # Set the font to be serif, rather than sans
    sns.set(font="serif")
    # Make the background white, and specify the
    # specific font family
    sns.set_style(
        "white",
        {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
    sns.set_style("ticks")
    sns.set_style("whitegrid")


def plot_history(history, model_name):
    plt.clf()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Autoencoder Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    # plt.show()
    plt.savefig(stt.TRAINING_CURVES_PATH + "/" + model_name + ".png", format="png")


def plot_scores(
    positive_scores,
    negative_scores,
    filename="scores.png",
    title="Score distribution",
):
    set_style()
    plt.clf()
    sns.distplot(positive_scores, norm_hist=True, color="green", bins=50)
    sns.distplot(negative_scores, norm_hist=True, color="red", bins=50)
    # plt.legend(loc='upper left')
    plt.legend(["Genuine", "Impostor"], loc="best")
    plt.xlabel("Score")
    plt.title(title)
    # plt.show()
    plt.savefig("output_png/" + filename, format="png")


def plot_training(history, model_name, metrics="loss"):
    # list all data in history
    print(history.history.keys())
    keys = list(history.history.keys())
    plt.figure()
    if metrics == "loss":
        plt.plot(history.history[keys[0]])
        plt.plot(history.history[keys[2]])
        plt.title("Model loss " + model_name)
        plt.ylabel("loss")
    if metrics == "accuracy":
        plt.plot(history.history[keys[1]])
        plt.plot(history.history[keys[3]])
        # plt.plot(history.history['categorical_accuracy'])
        # plt.plot(history.history['val_categorical_accuracy'])
        plt.title("Model accuracy " + model_name)
        plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["training", "validation"], loc="upper left")
    # plt.show()
    plt.savefig(
        stt.TRAINING_CURVES_PATH + "/" + model_name + "_" + metrics,
        format="png",
    )


def plot_ROC_single(ee_file, title="ROC curve"):
    set_style()
    data = pd.read_csv(ee_file)
    auc = metrics.auc(data["FPR"], data["TPR"])
    plt.clf()
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")
    plt.plot(data["FPR"], data["TPR"], "-", label="AUC_EE = %0.2f" % auc)
    label = "AUC  = %0.2f" % auc
    legend_str = [label]
    plt.legend(legend_str)
    # plt.show()
    plt.savefig("roc.png")


def plot_ROC_filelist(
    filelist, labels, title="ROC curve", outputfilename="output_png/roc.png"
):
    # set_style()
    # font = {'family': 'normal', 'weight': 'bold', 'size': 12}
    font = {'family': 'normal', 'size': 16}
    plt.rc('font', **font)

    plt.clf()
    linestyles = [
        (0, (1, 10)),  # loosly dotted
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        "dashdot",
        "dashed",
        "dotted",
        "solid",
    ]

    colors = ["#1b9e77", "#d95f02", "#7570b3",  "#e7298a",  "#66a61e", "#e6ab02"]
    counter = 0
    for file in filelist:
        data = pd.read_csv(file)
        auc = metrics.auc(data["FPR"], data["TPR"])
        plt.plot(
            data["FPR"],
            data["TPR"],
            color=colors[counter],
            linestyle=linestyles[counter],
            label="AUC %s blocks = %0.2f" % (labels[counter], auc),
        )
        counter = counter + 1
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive rate")
    plt.legend(labels, loc="best")
    plt.savefig(outputfilename, type="png")
    # plt.show()



# df_original - dataframe 128 x 3 + userid
# df_generated - dataframe 128 x 2 + userid
def plot_curves_dx_dy_dt(df_original, df_generated, bezier=False):
    # df_original = normalize_rows(df_original)
    # df_generated = normalize_rows(df_generated)
    # startx,starty,stopx,stopy,length,userid
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for blank space between subplots
    hspace = 0.5  # the amount of height reserved for white space between subplots

    df_start = pd.read_csv("statistics/actions_start_stop_1min.csv")
    startx = df_start["startx"]
    starty = df_start["starty"]
    length = df_start["length"]
    print("original: " + str(df_original.shape))
    array1 = df_original.values
    print(array1.shape)
    nsamples, nfeatures = array1.shape
    nfeatures = nfeatures - 1
    X1 = array1[:, 0:nfeatures]

    print("generated: " + str(df_generated.shape))
    array2 = df_generated.values
    # array2 = array2.reshape(-1, stt.FEATURES * stt.DIMENSIONS, 1)
    X2 = array2
    input_size = stt.FEATURES
    # input_dim = 2
    for i in range(stt.NUM_PLOTS):
        plt.clf()
        plt.subplots_adjust(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            wspace=wspace,
            hspace=hspace,
        )
        # original
        dx1 = X1[i, 0:input_size]
        dy1 = X1[i, input_size:2 * input_size]
        dt1 = X1[i, 2 * input_size:3 * input_size]  # new Bezier - PCA
    # bezier_baseline = [0.81, 0.86, 0.90, 0.91, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96,]
    # bezier_humanlike = [0.61, 0.65, 0.68, 0.70, 0.71, 0.73, 0.73, 0.74, 0.75, 0.75,]
    # unsup_ae_cnn = [0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88,]
    # unsup_ae_rnn = [0.86, 0.92, 0.95, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99,]
    # sup_ae_cnn = [0.62, 0.64, 0.65, 0.65, 0.66, 0.66, 0.67, 0.67, 0.67, 0.67,]
    # sup_ae_rnn = [0.63, 0.64, 0.65, 0.66, 0.67, 0.67, 0.67, 0.68, 0.68, 0.68,]
        # generated
        dx2 = X2[i, 0:input_size]
        dy2 = X2[i, input_size:2 * input_size]
        # trajectory original
        x1 = [startx[i]]
        for e in dx1:
            x1.append(x1[-1] + e)
        y1 = [starty[i]]
        for e in dy1:
            y1.append(y1[-1] + e)
        t1 = [0]
        for e in dt1:
            t1.append(t1[-1] + e)
        # trajectory generated
        x2 = [startx[i]]
        for e in dx2:
            x2.append(x2[-1] + e)
        y2 = [starty[i]]
        for e in dy2:
            y2.append(y2[-1] + e)

        dx1 = dx1[0:length[i]]
        dx2 = dx2[0:length[i]]
        t1 = t1[0:length[i]]
        x1 = x1[0:length[i]]
        x2 = x2[0:length[i]]
        y1 = y1[0:length[i]]
        y2 = y2[0:length[i]]
        dy1 = dy1[0:length[i]]
        dy2 = dy2[0:length[i]]

        plt.subplot(3, 1, 1)
        # plt.scatter(x1, y1, color='blue', marker='o')
        plt.plot(x1, y1, color="blue", marker="o")
        plt.plot(x2, y2, color="red", marker=".")
        # plt.scatter(x2, y2, color='red', marker='.')
        plt.plot(startx[i], starty[i], "s")
        plt.title("trajectory")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.subplot(3, 1, 2)
        plt.plot(t1, dx1, color="blue", marker=".")
        plt.plot(t1, dx2, color="red", marker=".")
        # plt.title('dx')
        plt.xlabel("time (ms)")
        plt.ylabel("dx (pixel)")

        plt.subplot(3, 1, 3)
        plt.plot(t1, dy1, color="blue", marker=".")
        plt.plot(t1, dy2, color="red", marker=".")
        # plt.title('dy')
        plt.xlabel("time (ms)")
        plt.ylabel("dy (pixel)")

        plt.legend(["Human", "Generated"], loc="best")
        if bezier:
            plt.savefig("output_png/bezier" + "/temp" + str(i) + ".png")
        else:
            plt.savefig(stt.OUTPUT_PNG + "/temp" + str(i) + ".png")


# df_original - dataframe 128 x 3 + userid
def plot_original_curves_dx_dy_dt(df_original):
    # df_original = normalize_rows(df_original)
    # df_generated = normalize_rows(df_generated)
    # startx,starty,stopx,stopy,length,userid
    left = 0.125  # the left side of the subplots of the figure
    right = 0.9  # the right side of the subplots of the figure
    bottom = 0.1  # the bottom of the subplots of the figure
    top = 0.9  # the top of the subplots of the figure
    wspace = 0.5  # the amount of width reserved for blank space between subplots
    hspace = 0.5  # the amount of height reserved for white space between subplots

    df_start = pd.read_csv("statistics/actions_start_stop_1min.csv")
    startx = df_start["startx"]
    starty = df_start["starty"]
    length = df_start["length"]
    print("original: " + str(df_original.shape))
    array1 = df_original.values
    print(array1.shape)
    nsamples, nfeatures = array1.shape
    nfeatures = nfeatures - 1
    X1 = array1[:, 0:nfeatures]
    input_size = stt.FEATURES
    # input_dim = 2
    for i in range(stt.NUM_PLOTS):
        plt.clf()
        plt.subplots_adjust(
            left=left,
            bottom=bottom,
            right=right,
            top=top,
            wspace=wspace,
            hspace=hspace,
        )
        # original
        dx1 = X1[i, 0:input_size]
        dy1 = X1[i, input_size:2 * input_size]
        dt1 = X1[i, 2 * input_size:3 * input_size]
        # trajectory original
        x1 = [startx[i]]
        for e in dx1:
            x1.append(x1[-1] + e)
        y1 = [starty[i]]
        for e in dy1:
            y1.append(y1[-1] + e)
        t1 = [0]
        for e in dt1:
            t1.append(t1[-1] + e)
        t1 = t1[0:length[i]]
        x1 = x1[0:length[i]]
        y1 = y1[0:length[i]]
        dx1 = dx1[0:length[i]]
        dy1 = dy1[0:length[i]]
        
        plt.subplot(3, 1, 1)
        plt.plot(x1, y1, color="blue", marker="o")
        plt.plot(startx[i], starty[i], color="magenta", marker="s")
        plt.title("trajectory")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.subplot(3, 1, 2)
        plt.plot(t1, dx1, color="blue", marker=".")
        plt.xlabel("time (ms)")
        plt.ylabel("dx (pixel)")

        plt.subplot(3, 1, 3)
        plt.plot(t1, dy1, color="blue", marker=".")
        plt.xlabel("time (ms)")
        plt.ylabel("dy (pixel)")
        plt.savefig("output_png/human" + "/temp" + str(i) + ".png")


# Access - 2021. augusztus
def plot_multiple_auc():
    plt.clf()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # new Bezier - OCSVM
    # bezier_baseline = [0.82, 0.87, 0.89, 0.91, 0.93, 0.93, 0.94, 0.95, 0.95, 0.96]
    # bezier_humanlike = [0.60, 0.64, 0.67, 0.69, 0.70, 0.72, 0.73, 0.73, 0.74, 0.74]
    # unsup_ae_cnn = [0.75, 0.78, 0.80, 0.81, 0.82, 0.83, 0.83, 0.84, 0.84, 0.85]
    # unsup_ae_rnn = [0.89, 0.93, 0.96, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99]
    # sup_ae_cnn = [0.65, 0.66, 0.67, 0.67, 0.68, 0.68, 0.69, 0.69, 0.69, 0.69]
    # sup_ae_rnn = [0.68, 0.68, 0.69, 0.70, 0.70, 0.71, 0.71, 0.72, 0.72, 0.72]

    # new Bezier - LOF
    # bezier_baseline = [0.80, 0.88, 0.92, 0.95, 0.96, 0.97, 0.98, 0.98, 0.98, 0.99]
    # bezier_humanlike = [0.71, 0.78, 0.82, 0.85, 0.87, 0.88, 0.90, 0.90, 0.91, 0.92]
    # unsup_ae_cnn = [0.78, 0.86, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.97, 0.98]
    # unsup_ae_rnn = [0.97, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    # sup_ae_cnn = [0.57, 0.61, 0.63, 0.65, 0.67, 0.68, 0.70, 0.71, 0.72, 0.73]
    # sup_ae_rnn = [0.60, 0.64, 0.67, 0.69, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76]

    # new Bezier - Feature Bagging
    bezier_baseline = [0.80, 0.88, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.98, 0.98]
    bezier_humanlike = [0.72, 0.78, 0.82, 0.85, 0.86, 0.88, 0.89, 0.90, 0.90, 0.91,]
    unsup_ae_cnn = [0.80, 0.87, 0.90, 0.93, 0.94, 0.95, 0.96, 0.96, 0.97, 0.97,]
    unsup_ae_rnn = [0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    sup_ae_cnn = [0.59, 0.61, 0.63, 0.64, 0.66, 0.66, 0.67, 0.68, 0.69, 0.69]
    sup_ae_rnn = [0.62, 0.66, 0.69, 0.71, 0.73, 0.75, 0.76, 0.77, 0.78, 0.79]

    # new Bezier - PCA
    # bezier_baseline = [0.81, 0.86, 0.90, 0.91, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96,]
    # bezier_humanlike = [0.61, 0.65, 0.68, 0.70, 0.71, 0.73, 0.73, 0.74, 0.75, 0.75,]
    # unsup_ae_cnn = [0.75, 0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88,]
    # unsup_ae_rnn = [0.86, 0.92, 0.95, 0.97, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99,]
    # sup_ae_cnn = [0.62, 0.64, 0.65, 0.65, 0.66, 0.66, 0.67, 0.67, 0.67, 0.67,]
    # sup_ae_rnn = [0.63, 0.64, 0.65, 0.66, 0.67, 0.67, 0.67, 0.68, 0.68, 0.68,]

    linestyles = [
        (0, (1, 10)),  # loosly dotted
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        "dashdot",
        "dashed",
        "dotted",
        "solid",
    ]

    colors = ["#1b9e77", "#d95f02", "#7570b3",  "#e7298a",  "#66a61e", "#e6ab02"]

    labels = [
        "Bezier baseline",
        "Bezier humanlike",
        "CNN-AE conventional",
        "RNN-AE conventional",
        "CNN-AE our approach",
        "RNN-AE our approach",
    ]
    plt.plot(x, bezier_baseline, color=colors[0], linestyle=linestyles[0])
    plt.plot(x, bezier_humanlike, color=colors[1], linestyle=linestyles[1])
    plt.plot(x, unsup_ae_cnn, color=colors[2], linestyle=linestyles[2])
    plt.plot(x, unsup_ae_rnn, color=colors[3], linestyle=linestyles[3])
    plt.plot(x, sup_ae_cnn, color=colors[4], linestyle=linestyles[4])
    plt.plot(x, sup_ae_rnn, color=colors[5], linestyle=linestyles[5])
    plt.xlabel("Number of aggregated mouse trajectories")
    plt.ylabel("AUC")
   
    xlabels = [i for i in range(1, 11)]
    plt.xticks(x, xlabels)

    ylabels = np.arange(0.4, 1.01, 0.1)
    plt.yticks(ylabels)

    plt.legend(labels, loc="lower right")
    plt.savefig("output_png/auc_lineplot.png")


# Access - 2021. augusztus
def plot_multiple_eer():
    plt.clf()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  
    # new Bezier - OCSVM
    # bezier_baseline =  [0.26, 0.25, 0.21, 0.18, 0.16, 0.15, 0.13, 0.13, 0.12, 0.12]
    # bezier_humanlike = [0.43, 0.39, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.32, 0.31]
    # unsup_ae_cnn = [0.31, 0.30, 0.29, 0.27, 0.26, 0.25, 0.24, 0.23, 0.23, 0.23,]
    # unsup_ae_rnn = [0.24, 0.18, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04]
    # sup_ae_cnn = [0.38, 0.38, 0.38, 0.37, 0.37, 0.36, 0.36, 0.36, 0.36, 0.35,]
    # sup_ae_rnn = [0.37, 0.37, 0.37, 0.36, 0.36, 0.35, 0.35, 0.34, 0.34, 0.34]


    # new Bezier - LOF
    # bezier_baseline = [0.28, 0.19, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,]
    # bezier_humanlike = [0.35, 0.29, 0.25, 0.22, 0.21, 0.19, 0.18, 0.17, 0.17, 0.16]
    # unsup_ae_cnn = [0.30, 0.23, 0.18, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07,]
    # unsup_ae_rnn = [0.09, 0.04, 0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00,]
    # sup_ae_cnn = [0.45, 0.43, 0.41, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32,]
    # sup_ae_rnn = [0.44, 0.42, 0.39, 0.37, 0.35, 0.34, 0.33, 0.31, 0.31, 0.30,]
    
    # new Bezier - FeatureBagging
    bezier_baseline =  [0.28, 0.20, 0.16, 0.13, 0.11, 0.09, 0.08, 0.08, 0.07, 0.06]
    bezier_humanlike = [0.34, 0.28, 0.25, 0.23, 0.22, 0.20, 0.19, 0.18, 0.17, 0.17]
    unsup_ae_cnn = [0.28, 0.21, 0.17, 0.15, 0.13, 0.11, 0.10, 0.09, 0.08, 0.08]
    unsup_ae_rnn = [0.06, 0.03, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00]
    sup_ae_cnn = [0.45, 0.44, 0.42, 0.40, 0.39, 0.38, 0.37, 0.37, 0.36, 0.36]
    sup_ae_rnn = [0.42, 0.40, 0.37, 0.35, 0.34, 0.32, 0.31, 0.30, 0.29, 0.28]

    # new Bezier - PCA
    # bezier_baseline = [0.31, 0.24, 0.19, 0.17, 0.15, 0.14, 0.13, 0.12, 0.11, 0.11]
    # bezier_humanlike = [0.43, 0.39, 0.37, 0.35, 0.34, 0.33, 0.33, 0.32, 0.32, 0.31]
    # unsup_ae_cnn = [0.35, 0.28, 0.25, 0.23, 0.22, 0.21, 0.21, 0.20, 0.19, 0.19,]
    # unsup_ae_rnn = [0.29, 0.17, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04,]
    # sup_ae_cnn = [0.40, 0.39, 0.39, 0.39, 0.38, 0.37, 0.37, 0.37, 0.37, 0.37,]
    # sup_ae_rnn = [0.41, 0.40, 0.39, 0.39, 0.38, 0.38, 0.37, 0.37, 0.37, 0.37,]

    linestyles = [
        (0, (1, 10)),  # loosly dotted
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        "dashdot",
        "dashed",
        "dotted",
        "solid",
    ]
    colors = ["#1b9e77", "#d95f02", "#7570b3",  "#e7298a",  "#66a61e", "#e6ab02"]
    labels = [
        "Bezier baseline",
        "Bezier humanlike",
        "CNN-AE conventional",
        "RNN-AE conventional",
        "CNN-AE our approach",
        "RNN-AE our approach",
    ]

    plt.plot(x, bezier_baseline, color=colors[0], linestyle=linestyles[0])
    plt.plot(x, bezier_humanlike, color=colors[1], linestyle=linestyles[1])
    plt.plot(x, unsup_ae_cnn, color=colors[2], linestyle=linestyles[2])
    plt.plot(x, unsup_ae_rnn, color=colors[3], linestyle=linestyles[3])
    plt.plot(x, sup_ae_cnn, color=colors[4], linestyle=linestyles[4])
    plt.plot(x, sup_ae_rnn, color=colors[5], linestyle=linestyles[5])
    plt.xlabel("Number of aggregated mouse trajectories")
    plt.ylabel("EER")
    
    xlabels = [i for i in range(1, 11)]
    plt.xticks(x, xlabels)
    
    ylabels = np.arange(0.0, 0.75, 0.1)
    plt.yticks(ylabels)

    plt.legend(labels, loc="upper right")
    plt.savefig("output_png/eer_lineplot.png")


