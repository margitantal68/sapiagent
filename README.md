# sapiagent
SapiAgent: A Bot Based on DeepLearning to Generate Human-like MouseTrajectories

## Dataset
* SapiMouse - https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.html

## Code
### Folders

* bezier_actions - content will be generated
* equidistant actions - content will be generated
* output_png 
* output_roc_data 
* sapimouse - SapiMouse dataset - Download from here: https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.html
* bezier_actions - content will be generated
* sapimouse_actions - content will be generated
* statistics - endpoints and lengths of mouse actions (trajectories)
* TRAINED_MODELS
* TRAINING_CURVES

### Files

* anomaly_detection_pyod.py - anomaly detection evaluations using detectors from PyOD package
* autoencoder_models.py - CNN and RNN autoencoder models
* autoencoder_training.py - training autoencoders conventionally (unsupervised) or using our approach (supervised)
* create_bezier_actions.py - generate baseline and humanlike bezier actions
* create_equidistant_actions.py - generate the contents of the equidistant_actions folder 
* create_sapimouse_actions - generate the contents of the sapimouse_actions folder
* feature_extractions.py - extract meaningful features from actions (trajectories)
* generate_autoencoder_actions.py - generate actions (trajectories) using the trained autoencoder (type of autoencoder: settings.py); actions saved in generated_actions folder
* plots.py - plots
* settings.py - different configurations for running an experiment
* utils.py - utility functions

### Steps

We used [ML workspace](https://github.com/ml-tooling/ml-workspace) which is a web-based IDE for machine learning and 
data science (preloaded with popular data science libraries). 
Only the [pyclick](https://pypi.org/project/pyclick/) package was added to this workspace.

1. Download and unzip the SapiMouse dataset into **sapimouse** folder
2. Segment SapiMouse dataset into actions: **python create_sapimouse_actions.py**
3. Create Bezier **baseline** and **humanlike** datasets using the endpoints from SapMouse S1 (1 min session):  **python create_bezier_actions.py**
4. Create equidistant actions, that will be used for training the autoencoders (supervised): **python create_equidistant_actions.py** 
5. Train an autoencoder, then generate the corresponding actions. Use settings.py to set the desired architecture and training type.
    1. Set training parameters **settings.py** 
        1. CNN_AE, conventional training: TRAINING_TYPE = 'unsupervised', KEY ='fcn' 
        2. RNN_AE, conventional training: TRAINING_TYPE = 'unsupervised', KEY ='bidirectional' 
        3. CNN_AE, our approach: TRAINING_TYPE = 'supervised', KEY ='fcn' 
        4. RNN_AE, our approach: TRAINING_TYPE = 'supervised', KEY ='bidirectional' 
    2. Train the autoencoder: **python autoencoder_training.py**
    3. Generate actions (trajectories): **python generate_autoencoder_actions.py**
6. Evaluate the quality of the generated actions: **python anomaly_detection_pyod.py**


