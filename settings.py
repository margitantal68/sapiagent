OUTPUT_FIGURES = "output_png"
TRAINING_CURVES_PATH = "TRAINING_CURVES"
TRAINED_MODELS_PATH = "TRAINED_MODELS"

# Init random generator
RANDOM_STATE = 11235

# Model parameters
BATCH_SIZE = 32
EPOCHS = 100

# Temporary filename - used to save ROC curve data
TEMP_NAME = "scores.csv"

# Input shape MOUSE - SapiMouse
FEATURES = 128
DIMENSIONS = 2

# Other parameters
TRAINING = True
# 'fcn', 'bidirectional'
KEY = "fcn"
# 'mse' 'custom'
LOSS = "mse"

SUFFIX = "dx_dy"


# 'supervised', 'supervised'
TRAINING_TYPE = "unsupervised"
OUTPUT_PNG = OUTPUT_FIGURES + "/" + KEY + "_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE

# model names
model_names = {
    "fcn": "fcn_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE + ".h5",
    "bidirectional": "bidirectional_" + SUFFIX + "_" + LOSS + "_" + TRAINING_TYPE + ".h5",
}

# number of plots
NUM_PLOTS = 900

# anomaly_detection.py
# number of actions (trajectories) used for decision
NUM_ACTIONS = 10
