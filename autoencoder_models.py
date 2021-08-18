import time
import numpy as np
import pandas as pd
import tensorflow as tf
import settings as stt
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Conv1D,
    Conv1DTranspose,
    GlobalAveragePooling1D,
    Reshape,
    TimeDistributed,
    UpSampling1D,
)
from keras.layers import Bidirectional, GRU
from keras.models import Model, load_model
from tensorflow.keras import backend as K


def bidirectional_autoencoder(input_size, input_dim):
    input_layer = Input(shape=(input_size, input_dim))
    encoded = Bidirectional(GRU(16, return_sequences=True))(input_layer)
    encoded = Dropout(0.25)(encoded)
    decoded = Bidirectional(GRU(16, return_sequences=True))(encoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    if stt.LOSS == "custom":
        autoencoder.compile(optimizer="adam", loss=custom_loss)
    else:
        autoencoder.compile(optimizer="adam", loss=stt.LOSS)
    autoencoder.summary()
    return encoder, autoencoder


def fcn_autoencoder(input_shape, fcn_filters=128, bottleneck=True):
    input_layer = Input(shape=input_shape)
    conv1 = Conv1D(
        filters=fcn_filters, kernel_size=8, padding="same", activation="relu"
    )(input_layer)
    conv2 = Conv1D(
        filters=2 * fcn_filters,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(conv1)
    conv3 = Conv1D(
        filters=fcn_filters, kernel_size=3, padding="same", activation="relu"
    )(conv2)
    encoded = conv3
    h = conv3
    if bottleneck:
        encoded = GlobalAveragePooling1D()(conv3)
        dim_encoded = K.int_shape(encoded)[1]
        h = Reshape((dim_encoded, 1))(encoded)
        # stt.FEATURES must be multiple of 128
        factor = 1
        if dim_encoded < stt.FEATURES:
            factor = (int)(stt.FEATURES / dim_encoded)
        h = UpSampling1D(factor)(h)
    conv3 = Conv1DTranspose(
        filters=fcn_filters, kernel_size=3, padding="same", activation="relu"
    )(h)
    conv2 = Conv1DTranspose(
        filters=2 * fcn_filters,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(conv3)
    conv1 = Conv1DTranspose(
        filters=stt.DIMENSIONS, kernel_size=8, padding="same"
    )(conv2)
    decoded = conv1
    encoder = Model(input_layer, encoded)
    autoencoder = Model(input_layer, decoded)
    if stt.LOSS == "custom":
        autoencoder.compile(optimizer="adam", loss=custom_loss)
    else:
        autoencoder.compile(optimizer="adam", loss=stt.LOSS)
    autoencoder.summary()
    return encoder, autoencoder


def custom_loss(y_actual, y_predicted):
    result_mae = K.sum(K.abs(y_actual - y_predicted), axis=-1)
    actual_sum = K.sum(y_actual, axis=-1)
    predicted_sum = K.sum(y_predicted, axis=-1)
    return 0.25 * result_mae + 0.75 * K.abs(actual_sum - predicted_sum)
