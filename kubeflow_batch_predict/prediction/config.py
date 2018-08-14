from enum import Enum


class UserClassType(Enum):
    model_class = "model_class"
    processor_class = "processor_class"


ENGINE = "Prediction-Engine"
ENGINE_RUN_TIME = "Prediction-Engine-Run-Time"
FRAMEWORK = "Framework"
SCIKIT_LEARN_FRAMEWORK_NAME = "scikit_learn"
XGBOOST_FRAMEWORK_NAME = "xgboost"
TENSORFLOW_FRAMEWORK_NAME = "tensorflow"
PREPROCESS_TIME = "Prediction-Preprocess-Time"
POSTPROCESS_TIME = "Prediction-Postprocess-Time"
# Keys for the name of the methods that the user provided `Processor`
# class should implement.
PREPROCESS_KEY = "preprocess"
POSTPROCESS_KEY = "postprocess"
FROM_MODEL_KEY = "from_model_path"

# Additional TF keyword arguments
INPUTS_KEY = "inputs"
OUTPUTS_KEY = "outputs"
SIGNATURE_KEY = "signature_name"

# Stats
COLUMNARIZE_TIME = "Prediction-Columnarize-Time"
UNALIAS_TIME = "Prediction-Unalias-Time"
ENCODE_TIME = "Prediction-Encode-Time"
SESSION_RUN_TIME = "Prediction-Session-Run-Time"
ALIAS_TIME = "Prediction-Alias-Time"
ROWIFY_TIME = "Prediction-Rowify-Time"
SESSION_RUN_ENGINE_NAME = "TF_SESSION_RUN"

# Scikit-learn and XGBoost related constants
MODEL_FILE_NAME_JOBLIB = "model.joblib"
MODEL_FILE_NAME_PICKLE = "model.pkl"
MODEL_FILE_NAME_BST = "model.bst"
