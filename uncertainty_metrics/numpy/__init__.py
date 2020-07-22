# coding=utf-8
# Copyright 2020 The uncertainty_metrics Authors.
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

"""Uncertainty Metrics Numpy init file."""

# pylint: disable=g-import-not-at-top
try:
  import numpy as np  # pylint: disable=unused-import
except ImportError:
  pass
else:
  from uncertainty_metrics.numpy import visualization
  from uncertainty_metrics.numpy.censored_calibration_error import CCE
  from uncertainty_metrics.numpy.censored_calibration_error import cce
  from uncertainty_metrics.numpy.censored_calibration_error import cce_conf_int
  from uncertainty_metrics.numpy.general_calibration_error import *
  from uncertainty_metrics.numpy.general_calibration_error import ace
  from uncertainty_metrics.numpy.general_calibration_error import adaptive_calibration_error
  from uncertainty_metrics.numpy.general_calibration_error import ece
  from uncertainty_metrics.numpy.general_calibration_error import gce
  from uncertainty_metrics.numpy.general_calibration_error import general_calibration_error
  from uncertainty_metrics.numpy.general_calibration_error import GeneralCalibrationError
  from uncertainty_metrics.numpy.general_calibration_error import rmsce
  from uncertainty_metrics.numpy.general_calibration_error import root_mean_squared_calibration_error
  from uncertainty_metrics.numpy.general_calibration_error import sce
  from uncertainty_metrics.numpy.general_calibration_error import static_calibration_error
  from uncertainty_metrics.numpy.general_calibration_error import tace
  from uncertainty_metrics.numpy.general_calibration_error import thresholded_adaptive_calibration_error
  from uncertainty_metrics.numpy.visualization import reliability_diagram
  from uncertainty_metrics.numpy.semiparametric_calibration_error import semiparametric_calibration_error
  from uncertainty_metrics.numpy.semiparametric_calibration_error import semiparametric_calibration_error_conf_int
  from uncertainty_metrics.numpy.semiparametric_calibration_error import SemiparametricCalibrationError
  from uncertainty_metrics.numpy.semiparametric_calibration_error import SPCE
  from uncertainty_metrics.numpy.semiparametric_calibration_error import spce
  from uncertainty_metrics.numpy.semiparametric_calibration_error import spce_conf_int

  _allowed_symbols = [
      "reliability_diagram",
      "visualization",
      "GeneralCalibrationError",
      "general_calibration_error",
      "ece",
      "root_mean_squared_calibration_error",
      "rmsce",
      "static_calibration_error",
      "sce",
      "adaptive_calibration_error",
      "ace",
      "gce",
      "thresholded_adaptive_calibration_error",
      "tace",
      "semiparametric_calibration_error",
      "semiparametric_calibration_error_conf_int",
      "SemiparametricCalibrationError",
      "SPCE",
      "spce",
      "spce_conf_int",
      "CCE",
      "cce",
      "cce_conf_int",
  ]

  try:
    from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import
  except ImportError:
    __all__ = _allowed_symbols
  else:
    remove_undocumented(__name__, _allowed_symbols)
