import settings as stt
import pandas as pd
import numpy as np
import os
import time
from keras.models import load_model


if __name__ == "__main__":
    OUTPUT_DIR = 'generated_actions'
    try:
        os.mkdir(OUTPUT_DIR)
    except OSError:
        print('Directory %s already exists' % OUTPUT_DIR)
    else:
        print('Successfuly created the directory %s' % OUTPUT_DIR)
    model_name = stt.model_names[stt.KEY]

    tic = time.perf_counter()
    # load model
    autoencoder = load_model('TRAINED_MODELS/' + model_name, compile=False)
    # Generate mouse curves from fixed endpoints - equidistant sequence
    df0 = pd.read_csv("equidistant_actions/equidistant_1min.csv", header=None)
    array = df0.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1 
    X = array[:, 0:nfeatures]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)
    # generate actions using the autoencoder
    df_generated = autoencoder.predict(X)
    dim1, dim2, dim3 = df_generated.shape
    df_generated = df_generated.reshape(dim1, dim2 * dim3)
    df_generated = pd.DataFrame(data=df_generated)
    df_generated = df_generated.apply(np.round)
    # Fix negative zeros issue
    df_generated[df_generated == 0.] = 0.
    filename = OUTPUT_DIR + "/generated_" + stt.KEY + "_" + stt.SUFFIX + '_' + stt.LOSS + '_' + stt.TRAINING_TYPE + ".csv"
    df_generated.to_csv(filename, index=False, header=False)
    toc = time.perf_counter()

    print(f"Execution time: {toc - tic:0.4f} seconds")
