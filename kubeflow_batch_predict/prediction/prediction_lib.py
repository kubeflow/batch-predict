# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for running predictions.
"""
import base64
import collections
from contextlib import contextmanager
import inspect
import json
import logging
import os
import pickle
import pydoc  # used for importing python classes from their FQN
import sys
import timeit

from ._interfaces import Model
from ._interfaces import PredictionClient
from enum import Enum
import numpy as np
import six

from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat


# --------------------------
# prediction.prediction_lib
# --------------------------
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

PredictionErrorType = collections.namedtuple(
    "PredictionErrorType", ("message", "code"))


class PredictionError(Exception):
  """Customer exception for known prediction exception."""

  # The error code for prediction.
  FAILED_TO_LOAD_MODEL = PredictionErrorType(
      message="Failed to load model", code=0)
  INVALID_INPUTS = PredictionErrorType("Invalid inputs", code=1)
  FAILED_TO_RUN_MODEL = PredictionErrorType(
      message="Failed to run the provided model", code=2)
  INVALID_OUTPUTS = PredictionErrorType(
      message="There was a problem processing the outputs", code=3)
  INVALID_USER_CODE = PredictionErrorType(
      message="There was a problem processing the user code", code=4)
  # When adding new exception, please update the ERROR_MESSAGE_ list as well as
  # unittest.

  def __init__(self, error_code, error_detail, *args):
    super(PredictionError, self).__init__(error_code, error_detail, *args)

  @property
  def error_code(self):
    return self.args[0].code

  @property
  def error_message(self):
    return self.args[0].message

  @property
  def error_detail(self):
    return self.args[1]

  def __str__(self):
    return ("%s: %s (Error code: %d)" % (self.error_message,
                                         self.error_detail, self.error_code))




class BaseModel(Model):
  """The base definition of an internal Model interface.
  """

  def __init__(self, client):
    """Constructs a BaseModel.

    Args:
      client: An instance of PredictionClient for performing prediction.
    """
    self._client = client
    self._user_processor = None

  def preprocess(self, instances, stats=None, **kwargs):
    """Runs the preprocessing function on the instances.

    Args:
      instances: list of instances as provided to the predict() method.
      stats: Stats object for recording timing information.
      **kwargs: Additional keyword arguments for preprocessing.

    Returns:
      A new list of preprocessed instances. Each instance is as described
      in the predict() method.
    """
    pass

  def postprocess(self, predicted_output, original_input=None, stats=None,
                  **kwargs):
    """Runs the postprocessing function on the instances.

    Args:
      predicted_output: list of instances returned by the predict() method on
        preprocessed instances.
      original_input: List of instances, before any pre-processing was applied.
      stats: Stats object for recording timing information.
      **kwargs: Additional keyword arguments for postprocessing.

    Returns:
      A new list of postprocessed instances.
    """
    pass

  def predict(self, instances, stats=None, **kwargs):
    """Runs preprocessing, predict, and postprocessing on the input."""

    stats = stats or Stats()
    self._validate_kwargs(kwargs)

    with stats.time(PREPROCESS_TIME):
      preprocessed = self.preprocess(instances, stats=stats, **kwargs)
    with stats.time(ENGINE_RUN_TIME):
      predicted_outputs = self._client.predict(
          preprocessed, stats=stats, **kwargs)
    with stats.time(POSTPROCESS_TIME):
      postprocessed = self.postprocess(
          predicted_outputs, original_input=instances, stats=stats, **kwargs)
    return instances, postprocessed

  def _validate_kwargs(self, kwargs):
    """Validates and sets defaults for extra predict keyword arguments.

    Modifies the keyword args dictionary in-place. Keyword args will be included
    into pre/post-processing and the client predict method.
    Can raise Exception to error out of request on bad keyword args.
    If no additional args are required, pass.

    Args:
      kwargs: Dictionary (str->str) of keyword arguments to check.
    """
    pass



def load_model_class(client, model_path):
  """Loads in the user specified custom Model class.

  Args:
    client: An instance of ModelServerClient for performing prediction.
    model_path: the path to either session_bundle or SavedModel

  Returns:
    An instance of a Model.
    Returns None if the user didn't specify the name of the custom
    python class to load in the create_version_request.

  Raises:
    PredictionError: for any of the following:
      (1) the user provided python model class cannot be found
      (2) if the loaded class does not implement the Model interface.
  """
  model_class = load_custom_class(UserClassType.model_class)
  if not model_class:
    return None
  model_instance = model_class.from_client(client, model_path)
  _validate_model_class(model_instance)
  return model_instance


def load_custom_class(class_type):
  """Loads in the user specified custom class.

  Args:
    class_type: An instance of UserClassType specifying what type of class to
    load.

  Returns:
    An instance of a class specified by the user in the `create_version_request`
    or None if no such class was specified.

  Raises:
    PredictionError: if the user provided python class cannot be found.
  """
  create_version_json = os.environ.get("create_version_request")
  if not create_version_json:
    return None
  create_version_request = json.loads(create_version_json)
  if not create_version_request:
    return None
  version = create_version_request.get("version")
  if not version:
    return None
  class_name = version.get(class_type.name)
  if not class_name:
    return None
  custom_class = pydoc.locate(class_name)
  # TODO(): right place to generate errors?
  if not custom_class:
    package_uris = [str(s) for s in version.get("package_uris")]
    raise PredictionError(PredictionError.INVALID_USER_CODE,
                          "%s cannot be found. Please make sure "
                          "(1) %s is the fully qualified function "
                          "name, and (2) %s uses the correct package "
                          "name as provided by the package_uris: %s" %
                          (class_name, class_type.name, class_type.name,
                           package_uris))
  return custom_class


def _validate_model_class(user_class):
  """Validates a user provided instance of a Model implementation.

  Args:
    user_class: An instance of a Model implementation.

  Raises:
    PredictionError: for any of the following:
      (1) the user model class does not have the correct method signatures for
      the predict method
  """
  user_class_name = type(user_class).__name__
  # Can't use isinstance() because the user doesn't have access to our Model
  # class. We can only inspect the user_class to check if it conforms to the
  # Model interface.
  if not hasattr(user_class, "predict"):
    raise PredictionError(PredictionError.INVALID_USER_CODE,
                          "The provided model class, %s, is missing the "
                          "required predict method." % user_class_name)
  # Check the predict method has the correct number of arguments
  user_signature = inspect.getargspec(user_class.predict)[0]
  model_signature = inspect.getargspec(Model.predict)[0]
  user_predict_num_args = len(user_signature)
  predict_num_args = len(model_signature)
  if predict_num_args is not user_predict_num_args:
    raise PredictionError(PredictionError.INVALID_USER_CODE,
                          "The provided model class, %s, has a predict method "
                          "with an invalid signature. Expected signature: %s "
                          "User signature: %s" %
                          (user_class_name, model_signature, user_signature))


# TODO(user): Make this generic so it can load any Processor class, not just
# from the create_version_request.
def _new_processor_class(model_path=None):
  user_processor_cls = load_custom_class(UserClassType.processor_class)
  if user_processor_cls:
    user_preprocess_fn = getattr(user_processor_cls, PREPROCESS_KEY, None)
    user_postprocess_fn = getattr(user_processor_cls, POSTPROCESS_KEY, None)
    user_from_model_path_fn = getattr(user_processor_cls, FROM_MODEL_KEY, None)

    _validate_fn_signature(user_preprocess_fn,
                           ["self", "instances"],
                           PREPROCESS_KEY, user_processor_cls.__name__)
    _validate_fn_signature(user_postprocess_fn,
                           ["self", "instances"],
                           POSTPROCESS_KEY, user_processor_cls.__name__)
    _validate_fn_signature(user_from_model_path_fn, ["cls", "model_path"],
                           FROM_MODEL_KEY, user_processor_cls.__name__)
    if user_from_model_path_fn:
      return user_from_model_path_fn(model_path)  # pylint: disable=not-callable
    # Call the constructor if no `from_model_path` method provided.
    return user_processor_cls()


def _validate_fn_signature(fn, required_arg_names, expected_fn_name, cls_name):
  if not fn:
    return
  if not callable(fn):
    raise PredictionError(
        PredictionError.INVALID_USER_CODE,
        "The provided %s function in the Processor class "
        "%s is not callable." % (expected_fn_name, cls_name))
  for arg in required_arg_names:
    if arg not in inspect.getargspec(fn).args:
      raise PredictionError(
          PredictionError.INVALID_USER_CODE,
          "The provided %s function in the Processor class "
          "has an invalid signature. It should take %s as arguments but "
          "takes %s" %
          (fn.__name__, required_arg_names, inspect.getargspec(fn).args))







def _load_joblib_or_pickle_model(model_path):
  """Loads either a .joblib or .pkl file from GCS or from local.

  Loads one of MODEL_FILE_NAME_JOBLIB or MODEL_FILE_NAME_PICKLE files if they
  exist. This is used for both sklearn and xgboost.

  Arguments:
    model_path: The path to the directory that contains the model file. This
      path can be either a local path or a GCS path.

  Raises:
    PredictionError: If there is a problem while loading the file.

  Returns:
    A loaded scikit-learn or xgboost predictor object or None if neither
    MODEL_FILE_NAME_JOBLIB nor MODEL_FILE_NAME_PICKLE files are found.
  """
  try:
    # If we put this at the top, we need to add a dependency to sklearn
    # anywhere that prediction_lib is called.
    from sklearn.externals import joblib  # pylint: disable=g-import-not-at-top
  except Exception as e:
    error_msg = "Could not import sklearn module."
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
  try:
    if file_io.file_exists(os.path.join(model_path, MODEL_FILE_NAME_JOBLIB)):
      model_file_name = os.path.join(model_path, MODEL_FILE_NAME_JOBLIB)
      logging.info("Loading model %s using joblib.", model_file_name)
      with file_io.FileIO(model_file_name, mode="rb") as f:
        return joblib.load(f)

    elif file_io.file_exists(os.path.join(model_path, MODEL_FILE_NAME_PICKLE)):
      model_file_name = os.path.join(model_path, MODEL_FILE_NAME_PICKLE)
      logging.info("Loading model %s using pickle.", model_file_name)
      with file_io.FileIO(model_file_name, "rb") as f:
        return pickle.loads(f.read())

    return None
  except Exception as e:
    error_msg = (
        "Could not load the model: {}. {}. Please make sure the model was "
        "exported using python {}. Otherwise, please specify the correct "
        "'python_version' parameter when deploying the model. Currently, "
        "'python_version' accepts 2.7 and 3.5."
    ).format(model_file_name, str(e), sys.version_info[0])
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)


_FRAMEWORK_TO_MODEL_MAP = {
    TENSORFLOW_FRAMEWORK_NAME: (TensorFlowModel, create_tf_session_client),
    SCIKIT_LEARN_FRAMEWORK_NAME: (SklearnModel, create_sklearn_client),
    XGBOOST_FRAMEWORK_NAME: (XGBoostModel, create_xgboost_client)
}


def create_model(client,
                 model_path,
                 framework=TENSORFLOW_FRAMEWORK_NAME,
                 **unused_kwargs):
  """Creates and returns the appropriate model.

  Creates and returns a Model if no user specified model is
  provided. Otherwise, the user specified model is imported, created, and
  returned.

  Args:
    client: An instance of PredictionClient for performing prediction.
    model_path: The path to the exported model (e.g. session_bundle or
      SavedModel)
    framework: The framework used to train the model.

  Returns:
    An instance of the appropriate model class.
  """
  if framework is TENSORFLOW_FRAMEWORK_NAME:
    logging.info("Importing tensorflow.contrib in create_model")
    import tensorflow.contrib  # pylint: disable=redefined-outer-name, unused-variable, g-import-not-at-top
  model_cls = _FRAMEWORK_TO_MODEL_MAP[framework][0]
  return (load_model_class(client, model_path) or
          model_cls(client))


def create_client(framework, model_path, tags):
  framework = framework or TENSORFLOW_FRAMEWORK_NAME
  create_client_fn = _FRAMEWORK_TO_MODEL_MAP[framework][1]
  return create_client_fn(model_path, tags)



def _encode_str_tensor(data, tensor_name):
  """Encodes tensor data of type string.

  Data is a bytes in python 3 and a string in python 2. Base 64 encode the data
  if the tensorname ends in '_bytes', otherwise convert data to a string.

  Args:
    data: Data of the tensor, type bytes in python 3, string in python 2.
    tensor_name: The corresponding name of the tensor.

  Returns:
    JSON-friendly encoded version of the data.
  """
  if isinstance(data, list):
    return [_encode_str_tensor(val, tensor_name) for val in data]
  if tensor_name.endswith("_bytes"):
    return {"b64": compat.as_text(base64.b64encode(data))}
  else:
    return compat.as_text(data)


def local_predict(
    model_dir=None,
    tags=(tag_constants.SERVING,),
    signature_name=None,
    instances=None,
    framework=TENSORFLOW_FRAMEWORK_NAME):
  """Run a prediction locally."""
  instances = decode_base64(instances)
  client = create_client(framework, model_dir, tags)
  model = create_model(client, model_dir, framework)
  _, predictions = model.predict(instances, signature_name=signature_name)
  return {"predictions": list(predictions)}
