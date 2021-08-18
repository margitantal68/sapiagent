#sapiagent
SapiAgent: A Bot Based on DeepLearning to Generate Human-like MouseTrajectories

#dataset
* [SapiMouse] - https://ms.sapientia.ro/~manyi/sapimouse/sapimouse.html

#code
* folders
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

* files
    * anomaly_detection.py - anomaly detection evaluations using detectors from PyOD package
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

* Steps
    1. Download and unzip the sapimouse dataset --> sapimouse folder
    2. python create_sapimouse_actions.py
    3. python create_bezier_actions.py
    4. python create_equidistant_actions.py 
    5. Train autoencoder, then generate actions; results: 4 files in generated_actions folder
        1. settings.py: set TRAINING_TYPE {'supervised', 'unsupervised'} and set KEY {'fcn', 'bidirectional'}
        2. python autoencoder_training.py
        3. python generate_autoencoder_actions.py
    6. Evaluate the quality of the generated actions: python anomaly_detection.py


