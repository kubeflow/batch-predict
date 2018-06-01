# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=g-import-not-at-top
"""Classes and methods for predictions on a trained machine learning model.
"""
from kubeflow_batch_predict.prediction._interfaces import Model, PredictionClient
from kubeflow_batch_predict.prediction.base import BaseModel, canonicalize_single_tensor_input, rowify
from kubeflow_batch_predict.prediction.config import COLUMNARIZE_TIME, ENGINE, FRAMEWORK, \
    INPUTS_KEY, OUTPUTS_KEY, ROWIFY_TIME, SCIKIT_LEARN_FRAMEWORK_NAME, \
    SESSION_RUN_ENGINE_NAME, SESSION_RUN_TIME, TENSORFLOW_FRAMEWORK_NAME, XGBOOST_FRAMEWORK_NAME
from kubeflow_batch_predict.prediction.prediction_lib import create_client, create_model, local_predict
from kubeflow_batch_predict.prediction.sklearn_lib import create_sklearn_model, SklearnModel
from kubeflow_batch_predict.prediction.tensorflow_lib import load_model, SessionClient, TensorFlowClient, \
    TensorFlowModel
from kubeflow_batch_predict.prediction.utils import columnarize, decode_base64, encode_base64, Stats
from kubeflow_batch_predict.prediction.xgboost_lib import create_xgboost_model, XGBoostModel