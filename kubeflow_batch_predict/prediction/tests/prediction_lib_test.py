from mock import mock

from kubeflow_batch_predict.prediction._interfaces import PredictionClient, Model
from kubeflow_batch_predict.prediction.prediction_lib import _FRAMEWORK_TO_MODEL_MAP


def test_framework_to_model_map_all_implementing_model_interface():
    for _, (model, _) in _FRAMEWORK_TO_MODEL_MAP.items():
        assert issubclass(model, Model)
