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

"""Uncertainty Metrics with Tensorflow backend."""

from uncertainty_metrics.tensorflow import accuracy
from uncertainty_metrics.tensorflow import calibration
from uncertainty_metrics.tensorflow import diversity
from uncertainty_metrics.tensorflow import information_criteria
from uncertainty_metrics.tensorflow import scoring_rules
from uncertainty_metrics.tensorflow.accuracy import OracleCollaborativeAccuracy
from uncertainty_metrics.tensorflow.auc import AUC
from uncertainty_metrics.tensorflow.calibration import bayesian_expected_calibration_error
from uncertainty_metrics.tensorflow.calibration import ExpectedCalibrationError
from uncertainty_metrics.tensorflow.diversity import average_pairwise_diversity
from uncertainty_metrics.tensorflow.diversity import cosine_distance
from uncertainty_metrics.tensorflow.diversity import disagreement
from uncertainty_metrics.tensorflow.diversity import double_fault
from uncertainty_metrics.tensorflow.diversity import kl_divergence
from uncertainty_metrics.tensorflow.diversity import logit_kl_divergence
from uncertainty_metrics.tensorflow.diversity import lp_distance
from uncertainty_metrics.tensorflow.diversity import variance_bound
from uncertainty_metrics.tensorflow.information_criteria import ensemble_cross_entropy
from uncertainty_metrics.tensorflow.information_criteria import gibbs_cross_entropy
from uncertainty_metrics.tensorflow.information_criteria import importance_sampling_cross_validation
from uncertainty_metrics.tensorflow.information_criteria import model_uncertainty
from uncertainty_metrics.tensorflow.information_criteria import negative_waic
from uncertainty_metrics.tensorflow.scoring_rules import brier_decomposition
from uncertainty_metrics.tensorflow.scoring_rules import brier_score
from uncertainty_metrics.tensorflow.scoring_rules import crps_normal_score
from uncertainty_metrics.tensorflow.scoring_rules import crps_score
from uncertainty_metrics.version import __version__
from uncertainty_metrics.version import VERSION

__all__ = [
    "accuracy",
    "average_pairwise_diversity",
    "AUC",
    "bayesian_expected_calibration_error",
    "brier_decomposition",
    "brier_score",
    "calibration",
    "cosine_distance",
    "crps_normal_score",
    "crps_score",
    "disagreement",
    "double_fault",
    "diversity",
    "ensemble_cross_entropy",
    "ExpectedCalibrationError",
    "gibbs_cross_entropy",
    "importance_sampling_cross_validation",
    "information_criteria",
    "kl_divergence",
    "logit_kl_divergence",
    "lp_distance",
    "model_uncertainty",
    "negative_waic",
    "OracleCollaborativeAccuracy",
    "scoring_rules",
    "variance_bound",
    "__version__",
    "VERSION",
]
