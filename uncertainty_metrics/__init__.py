# coding=utf-8
# Copyright 2020 The Uncertainty Metrics Authors.
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

"""Uncertainty Metrics."""

import warnings
# pylint: disable=g-import-not-at-top
try:
  from uncertainty_metrics import numpy
  __all__ = ["numpy"]
except ImportError:
  __all__ = []
  warnings.warn("NumPy backend for Uncertainty Merics is not available.")

try:
  from uncertainty_metrics import tensorflow
  from uncertainty_metrics.tensorflow import *
  # By default, `import uncertainty_metrics as um` uses the TensorFlow backend's
  # namespace.
  __all__ += tensorflow.__all__ + ["tensorflow"]
except ImportError:
  warnings.warn("TensorFlow backend for Uncertainty Merics is not available.")

try:
  numpy
except NameError:
  try:
    tensorflow
  except NameError:
    raise ImportError("Neither NumPy nor TensorFlow backends are available for "
                      "Uncertainty Metrics. Please install the dependencies "
                      "for either of them.")
# pylint: enable=g-import-not-at-top
