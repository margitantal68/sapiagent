
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import settings as stt
from sklearn.model_selection import train_test_split
from plots import plot_history
from autoencoder_models import bidirectional_autoencoder, fcn_autoencoder


def train_autoencoder(df, input_size, input_dim, model_name, num_filters):
    # split dataframe
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]

    # equidistant segments
    if stt.TRAINING_TYPE == 'supervised':
        df_synth = pd.read_csv("equidistant_actions/equidistant_3min.csv", header=None)
        X_synth = df_synth.values[:, :-1]

    X = X.reshape(-1, input_size, input_dim)
    if stt.TRAINING_TYPE == 'supervised':
        X_synth = X_synth.reshape(-1, input_size, input_dim)
        X_train, X_val, y_train, y_val, X_train_synth, X_val_synth, = train_test_split(X, y, X_synth, test_size=0.25, random_state=stt.RANDOM_STATE)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=stt.RANDOM_STATE)

    mini_batch_size = int(min(X_train.shape[0] / 10, stt.BATCH_SIZE))

    # convert to tensorflow dataset
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)
    if stt.TRAINING_TYPE == 'supervised':
        X_synth = np.asarray(X_synth).astype(np.float32)

    if stt.TRAINING_TYPE == 'supervised':
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_synth, X_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val_synth, X_val))
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    start_time = time.time()
    if stt.KEY == "fcn":
        encoder, model = fcn_autoencoder((input_size, input_dim), num_filters)
    if stt.KEY == "bidirectional":
        encoder, model = bidirectional_autoencoder(input_size, input_dim)
    # train model
    history = model.fit(train_ds, epochs=stt.EPOCHS, shuffle=False, validation_data=val_ds)
    duration = time.time() - start_time
    print("Training duration: " + str(duration / 60))
    print(model_name)
    plot_history(history, model_name)
    model.save(stt.TRAINED_MODELS_PATH + "/" + model_name)
    return encoder, model


if __name__ == "__main__":
    TRAINING_CURVES_PATH = "TRAINING_CURVES"
    TRAINED_MODELS_PATH = "TRAINED_MODELS"
    try:
        os.mkdir(TRAINING_CURVES_PATH)
        os.mkdir(TRAINED_MODELS_PATH)
    except OSError:
        print('Model will be saved in folder ' + TRAINED_MODELS_PATH)
        print('Training curve plot will be saved in folder ' + TRAINING_CURVES_PATH)
    model_name = stt.model_names[stt.KEY]
    if stt.TRAINING:
        # training data
        df_train = pd.read_csv("sapimouse_actions/actions_3min_dx_dy.csv", header=None)
        encoder, autoencoder = train_autoencoder(df_train, stt.FEATURES, stt.DIMENSIONS, model_name, num_filters=128)
    else:
        print('Set TRAINING to True in settings.py!')
