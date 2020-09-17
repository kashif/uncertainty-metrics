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

"""Uncertainty Metrics with NumPy backend."""

from uncertainty_metrics.numpy import visualization
from uncertainty_metrics.numpy.censored_calibration_error import CCE
from uncertainty_metrics.numpy.censored_calibration_error import cce
from uncertainty_metrics.numpy.censored_calibration_error import cce_conf_int
from uncertainty_metrics.numpy.general_calibration_error import ace
from uncertainty_metrics.numpy.general_calibration_error import adaptive_calibration_error
from uncertainty_metrics.numpy.general_calibration_error import compute_all_metrics
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
from uncertainty_metrics.numpy.semiparametric_calibration_error import semiparametric_calibration_error
from uncertainty_metrics.numpy.semiparametric_calibration_error import semiparametric_calibration_error_conf_int
from uncertainty_metrics.numpy.semiparametric_calibration_error import SemiparametricCalibrationError
from uncertainty_metrics.numpy.semiparametric_calibration_error import SPCE
from uncertainty_metrics.numpy.semiparametric_calibration_error import spce
from uncertainty_metrics.numpy.semiparametric_calibration_error import spce_conf_int
from uncertainty_metrics.numpy.visualization import plot_confidence_vs_accuracy_diagram
from uncertainty_metrics.numpy.visualization import plot_rejection_classification_diagram
from uncertainty_metrics.numpy.visualization import reliability_diagram
from uncertainty_metrics.version import __version__
from uncertainty_metrics.version import VERSION

__all__ = [
    "ace",
    "adaptive_calibration_error",
    "CCE",
    "cce",
    "cce_conf_int",
    "compute_all_metrics",
    "ece",
    "gce",
    "general_calibration_error",
    "GeneralCalibrationError",
    "plot_confidence_vs_accuracy_diagram",
    "plot_rejection_classification_diagram",
    "reliability_diagram",
    "rmsce",
    "root_mean_squared_calibration_error",
    "sce",
    "semiparametric_calibration_error",
    "semiparametric_calibration_error_conf_int",
    "SemiparametricCalibrationError",
    "SPCE",
    "spce",
    "spce_conf_int",
    "static_calibration_error",
    "tace",
    "thresholded_adaptive_calibration_error",
    "visualization",
    "__version__",
    "VERSION",
]
