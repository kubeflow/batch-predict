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
