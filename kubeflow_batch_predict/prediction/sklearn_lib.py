
# This class is specific to Scikit-learn, and should be moved to a separate
# module. However due to gcloud's complicated copying mechanism we need to keep
# things in one file for now.
class SklearnClient(PredictionClient):
  """A loaded scikit-learn model to be used for prediction."""

  def __init__(self, predictor):
    self._predictor = predictor

  def predict(self, inputs, stats=None, **kwargs):
    stats = stats or Stats()
    stats[FRAMEWORK] = SCIKIT_LEARN_FRAMEWORK_NAME
    stats[ENGINE] = SCIKIT_LEARN_FRAMEWORK_NAME
    try:
      return self._predictor.predict(inputs, **kwargs)
    except Exception as e:
      logging.exception("Exception while predicting with sklearn model.")
      raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL,
                            "Exception during sklearn prediction: " + str(e))

# (TODO) Move this to a separate Scikit-learn specific library.
class SklearnModel(BaseModel):
  """The implementation of Scikit-learn Model.
  """

  def __init__(self, client):
    super(SklearnModel, self).__init__(client)
    self._user_processor = _new_processor_class()
    if self._user_processor and hasattr(self._user_processor, PREPROCESS_KEY):
      self._preprocess = self._user_processor.preprocess
    else:
      self._preprocess = self._null_processor
    if self._user_processor and hasattr(self._user_processor, POSTPROCESS_KEY):
      self._postprocess = self._user_processor.postprocess
    else:
      self._postprocess = self._null_processor

  def predict(self, instances, stats=None, **kwargs):
    """Override the predict method to remove TF-specific args from kwargs."""
    kwargs.pop(SIGNATURE_KEY, None)
    return super(SklearnModel, self).predict(instances, stats, **kwargs)

  def preprocess(self, instances, stats=None, **kwargs):
    # TODO() Consider changing this to a more generic type.
    return self._preprocess(np.array(instances), **kwargs)

  def postprocess(self, predicted_outputs, original_input=None, stats=None,
                  **kwargs):
    # TODO() Consider changing this to a more generic type.
    post_processed = self._postprocess(predicted_outputs, **kwargs)
    if isinstance(post_processed, np.ndarray):
      return post_processed.tolist()
    if isinstance(post_processed, list):
      return post_processed
    raise PredictionError(
        PredictionError.INVALID_OUTPUTS,
        "Bad output type returned after running %s"
        "The post-processing function should return either "
        "a numpy ndarray or a list."
        % self._postprocess.__name__)

  def _null_processor(self, instances, **unused_kwargs):
    return instances



def create_sklearn_model(model_path, unused_flags):
  """Returns a sklearn model from the given model_path."""
  return SklearnModel(create_sklearn_client(model_path, None))

def create_sklearn_client(model_path, unused_tags):
  """Returns a prediction client for the corresponding sklearn model."""
  logging.info("Loading the scikit-learn model file from %s", model_path)
  sklearn_predictor = _load_joblib_or_pickle_model(model_path)
  if not sklearn_predictor:
    error_msg = "Could not find either {} or {} in {}".format(
        MODEL_FILE_NAME_JOBLIB, MODEL_FILE_NAME_PICKLE, model_path)
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
  # Check if the loaded python object is an sklearn model/pipeline.
  # Ex. type(sklearn_predictor).__module__ -> 'sklearn.svm.classes'
  #     type(pipeline).__module__ -> 'sklearn.pipeline'
  if "sklearn" not in type(sklearn_predictor).__module__:
    error_msg = ("Invalid model type detected: {}.{}. Please make sure the "
                 "model file is an exported sklearn model or pipeline.").format(
                     type(sklearn_predictor).__module__,
                     type(sklearn_predictor).__name__)
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)

  return SklearnClient(sklearn_predictor)

