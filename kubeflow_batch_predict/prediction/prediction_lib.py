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
import logging

from tensorflow.python.saved_model import tag_constants

from kubeflow_batch_predict.prediction import config, utils
from kubeflow_batch_predict.prediction.base import load_model_class
from kubeflow_batch_predict.prediction.sklearn_lib import SklearnModel, create_sklearn_client
from kubeflow_batch_predict.prediction.tensorflow_lib import TensorFlowModel, create_tf_session_client
from kubeflow_batch_predict.prediction.xgboost_lib import XGBoostModel, create_xgboost_client

_FRAMEWORK_TO_MODEL_MAP = {
    config.TENSORFLOW_FRAMEWORK_NAME: (TensorFlowModel, create_tf_session_client),
    config.SCIKIT_LEARN_FRAMEWORK_NAME: (SklearnModel, create_sklearn_client),
    config.XGBOOST_FRAMEWORK_NAME: (XGBoostModel, create_xgboost_client)
}


def create_model(client,
                 model_path,
                 framework=config.TENSORFLOW_FRAMEWORK_NAME,
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
    if framework is config.TENSORFLOW_FRAMEWORK_NAME:
        logging.info("Importing tensorflow.contrib in create_model")
    model_cls = _FRAMEWORK_TO_MODEL_MAP[framework][0]
    return (load_model_class(client, model_path) or
            model_cls(client))


def create_client(framework, model_path, tags):
    framework = framework or config.TENSORFLOW_FRAMEWORK_NAME
    create_client_fn = _FRAMEWORK_TO_MODEL_MAP[framework][1]
    return create_client_fn(model_path, tags)


def local_predict(
        model_dir=None,
        tags=(tag_constants.SERVING,),
        signature_name=None,
        instances=None,
        framework=config.TENSORFLOW_FRAMEWORK_NAME):
    """Run a prediction locally."""
    instances = utils.decode_base64(instances)
    client = create_client(framework, model_dir, tags)
    model = create_model(client, model_dir, framework)
    _, predictions = model.predict(instances, signature_name=signature_name)
    return {"predictions": list(predictions)}
