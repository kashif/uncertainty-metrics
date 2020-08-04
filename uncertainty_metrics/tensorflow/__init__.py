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

"""Namespace management for Tensorflow dependent code.
"""

# Make the TensorFlow backend be optional. The namespace is empty if
# TensorFlow is not available.
# pylint: disable=g-import-not-at-top
try:  # pylint: disable=g-statement-before-imports
  import tensorflow as tf  # pylint: disable=unused-import
  import tensorflow_probability as tfp
except ImportError:
  pass
else:
  from uncertainty_metrics.tensorflow.auc import AUC
  from uncertainty_metrics.tensorflow import calibration
  from uncertainty_metrics.tensorflow.calibration import brier_decomposition
  from uncertainty_metrics.tensorflow.calibration import expected_calibration_error
  from uncertainty_metrics.tensorflow.calibration import brier_score
  from uncertainty_metrics.tensorflow.calibration import bayesian_expected_calibration_error
  from uncertainty_metrics.tensorflow import mutual_information
  from uncertainty_metrics.tensorflow.mutual_information import model_uncertainty
  from uncertainty_metrics.tensorflow import posterior_predictive_criteria
  from uncertainty_metrics.tensorflow.posterior_predictive_criteria import negative_waic
  from uncertainty_metrics.tensorflow.posterior_predictive_criteria import importance_sampling_cross_validation
  from uncertainty_metrics.tensorflow import regression
  from uncertainty_metrics.tensorflow.regression import crps_score
  from uncertainty_metrics.tensorflow import metrics
  from uncertainty_metrics.tensorflow.metrics import ExpectedCalibrationError

  from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=g-direct-tensorflow-import

  _allowed_symbols = [
      "AUC",
      "calibration",
      "brier_decomposition",
      "expected_calibration_error",
      "brier_score",
      "bayesian_expected_calibration_error",
      "mutual_information",
      "model_uncertainty",
      "posterior_predictive_criteria",
      "negative_waic",
      "importance_sampling_cross_validation",
      "regression",
      "crps_score",
      "metrics",
      "ExpectedCalibrationError",
      "__version__",
      "VERSION",
  ]

  remove_undocumented(__name__, _allowed_symbols)
