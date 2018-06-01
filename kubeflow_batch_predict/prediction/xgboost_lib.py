
# (TODO) This class is specific to Xgboost, and should be moved to a
# separate module. However due to gcloud's complicated copying mechanism we need
# to keep things in one file for now.
class XgboostClient(PredictionClient):
  """A loaded xgboost model to be used for prediction."""

  def __init__(self, booster):
    self._booster = booster

  def predict(self, inputs, stats=None, **kwargs):
    stats = stats or Stats()
    stats[FRAMEWORK] = XGBOOST_FRAMEWORK_NAME
    stats[ENGINE] = XGBOOST_FRAMEWORK_NAME
    # TODO(): Move this to the top once b/64574886 is resolved.
    # Before then, it would work in production since we install xgboost in
    # the Dockerfile, but the problem is the unit test that will fail to build
    # and run since xgboost can not be added as a dependency to this target.
    import xgboost as xgb  # pylint: disable=g-import-not-at-top
    try:
      inputs_dmatrix = xgb.DMatrix(inputs)
    except Exception as e:
      logging.exception("Could not initialize DMatrix from inputs: ")
      raise PredictionError(
          PredictionError.FAILED_TO_RUN_MODEL,
          "Could not initialize DMatrix from inputs: " + str(e))
    try:
      return self._booster.predict(inputs_dmatrix, **kwargs)
    except Exception as e:
      logging.exception("Exception during predicting with xgboost model: ")
      raise PredictionError(PredictionError.FAILED_TO_RUN_MODEL,
                            "Exception during xgboost prediction: " + str(e))



# (TODO)Move this to a XGboost specific library.
class XGBoostModel(SklearnModel):
  """The implementation of XGboost Model.
  """

  def __init__(self, client):
    super(XGBoostModel, self).__init__(client)

def create_xgboost_client(model_path, unused_tags):
  """Returns a prediction client for the corresponding xgboost model."""
  logging.info("Loading the xgboost model from %s", model_path)
  booster = _load_joblib_or_pickle_model(model_path) or _load_xgboost_model(
      model_path)
  if not booster:
    error_msg = "Could not find {}, {}, or {} in {}".format(
        MODEL_FILE_NAME_JOBLIB, MODEL_FILE_NAME_PICKLE, MODEL_FILE_NAME_BST,
        model_path)
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
  # Check if the loaded python object is an xgboost model.
  # Expect type(booster).__module__ -> 'xgboost.core'
  if "xgboost" not in type(booster).__module__:
    error_msg = ("Invalid model type detected: {}.{}. Please make sure the "
                 "model file is an exported xgboost model.").format(
                     type(booster).__module__,
                     type(booster).__name__)
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)

  return XgboostClient(booster)

def _load_xgboost_model(model_path):
  """Loads an xgboost model from GCS or local.

  Args:
      model_path: path to the directory containing the xgboost model.bst file.
        This path can be either a local path or a GCS path.

  Returns:
    A xgboost.Booster with the model at model_path loaded.

  Raises:
    PredictionError: If there is a problem while loading the file.
  """
  # TODO(): Move this to the top once b/64574886 is resolved. Before
  # then, it would work in production since we install xgboost in the
  # Dockerfile, but the problem is the unit test that will fail to build and run
  # since xgboost can not be added as a dependency to this target.
  import xgboost as xgb  # pylint: disable=g-import-not-at-top
  model_file = os.path.join(model_path, MODEL_FILE_NAME_BST)
  if not file_io.file_exists(model_file):
    return None
  try:
    if model_file.startswith("gs://"):
      with file_io.FileIO(model_file, mode="rb") as f:
        # TODO(): Load model in memory twice. Use readinto if/when
        # that becomes available in FileIO. Or copy model locally before
        # loading.
        model_buf = bytearray(f.read())
        return xgb.Booster(model_file=model_buf)
    else:
      return xgb.Booster(model_file=model_file)
  except xgb.core.XGBoostError as e:
    error_msg = "Could not load the model: {}. {}.".format(
        os.path.join(model_path, MODEL_FILE_NAME_BST), str(e))
    logging.critical(error_msg)
    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)


def create_xgboost_model(model_path, unused_flags):
  """Returns a xgboost model from the given model_path."""
  return XGBoostModel(create_xgboost_client(model_path, None))


