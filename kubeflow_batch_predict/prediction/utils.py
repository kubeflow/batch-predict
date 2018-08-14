import base64
import collections
import timeit
from contextlib import contextmanager

import six
from tensorflow.python.framework import dtypes
from tensorflow.python.util import compat


MICRO = 1000000
MILLI = 1000


class Timer(object):
    """Context manager for timing code blocks.

    The object is intended to be used solely as a context manager and not
    as a general purpose object.

    The timer starts when __enter__ is invoked on the context manager
    and stopped when __exit__ is invoked. After __exit__ is called,
    the duration properties report the amount of time between
    __enter__ and __exit__ and thus do not change. However, if any of the
    duration properties are called between the call to __enter__ and __exit__,
    then they will return the "live" value of the timer.

    If the same Timer object is re-used in multiple with statements, the values
    reported will reflect the latest call. Do not use the same Timer object in
    nested with blocks with the same Timer context manager.

    Example usage:

      with Timer() as timer:
        foo()
      print(timer.duration_secs)
    """

    def __init__(self, timer_fn=None):
        self.start = None
        self.end = None
        self._get_time = timer_fn or timeit.default_timer

    def __enter__(self):
        self.end = None
        self.start = self._get_time()
        return self

    def __exit__(self, exc_type, value, traceback):
        self.end = self._get_time()
        return False

    @property
    def seconds(self):
        now = self._get_time()
        return (self.end or now) - (self.start or now)

    @property
    def microseconds(self):
        return int(MICRO * self.seconds)

    @property
    def milliseconds(self):
        return int(MILLI * self.seconds)


class Stats(dict):
    """An object for tracking stats.

    This class is dict-like, so stats are accessed/stored like so:

      stats = Stats()
      stats["count"] = 1
      stats["foo"] = "bar"

    This class also facilitates collecting timing information via the
    context manager obtained using the "time" method. Reported timings
    are in microseconds.

    Example usage:

      with stats.time("foo_time"):
        foo()
      print(stats["foo_time"])
    """

    @contextmanager
    def time(self, name, timer_fn=None):
        with Timer(timer_fn) as timer:
            yield timer
        self[name] = timer.microseconds


def columnarize(instances):
    """Columnarize inputs.

    Each line in the input is a dictionary of input names to the value
    for that input (a single instance). For each input "column", this method
    appends each of the input values to a list. The result is a dict mapping
    input names to a batch of input data. This can be directly used as the
    feed dict during prediction.

    For example,

      instances = [{"a": [1.0, 2.0], "b": "a"},
                   {"a": [3.0, 4.0], "b": "c"},
                   {"a": [5.0, 6.0], "b": "e"},]
      batch = prediction_server_lib.columnarize(instances)
      assert batch == {"a": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                       "b": ["a", "c", "e"]}

    Arguments:
      instances: (list of dict) where the dictionaries map input names
        to the values for those inputs.

    Returns:
      A dictionary mapping input names to values, as described above.
    """
    columns = collections.defaultdict(list)
    for instance in instances:
        for k, v in six.iteritems(instance):
            columns[k].append(v)
    return columns



def decode_base64(data):
    if isinstance(data, list):
        return [decode_base64(val) for val in data]
    elif isinstance(data, dict):
        if six.viewkeys(data) == {"b64"}:
            return base64.b64decode(data["b64"])
        else:
            return {k: decode_base64(v) for k, v in six.iteritems(data)}
    else:
        return data


def encode_base64(instances, outputs_map):
    """Encodes binary data in a JSON-friendly way."""
    if not isinstance(instances, list):
        raise ValueError("only lists allowed in output; got %s" %
                         (type(instances),))

    if not instances:
        return instances
    first_value = instances[0]
    if not isinstance(first_value, dict):
        if len(outputs_map) != 1:
            return ValueError("The first instance was a string, but there are "
                              "more than one output tensor, so dict expected.")
        # Only string tensors whose name ends in _bytes needs encoding.
        tensor_name, tensor_info = outputs_map.items()[0]
        tensor_type = tensor_info.dtype
        if tensor_type == dtypes.string:
            instances = _encode_str_tensor(instances, tensor_name)
        return instances

    encoded_data = []
    for instance in instances:
        encoded_instance = {}
        for tensor_name, tensor_info in six.iteritems(outputs_map):
            tensor_type = tensor_info.dtype
            tensor_data = instance[tensor_name]
            if tensor_type == dtypes.string:
                tensor_data = _encode_str_tensor(tensor_data, tensor_name)
            encoded_instance[tensor_name] = tensor_data
        encoded_data.append(encoded_instance)
    return encoded_data


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
