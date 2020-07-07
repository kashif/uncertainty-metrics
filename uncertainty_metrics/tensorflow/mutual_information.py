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

"""Mutual information metrics for probabilistic predictions of an ensemble.

Uncertainty in predictions due to model uncertainty can be assessed via
measures of the spread, or `disagreement` of an ensemble.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf  # tf


def model_uncertainty(logits):
  """Mutual information between the categorical label and the model parameters.

  A way to evaluate uncertainty in ensemble models is to measure its spread or
  `disagreement`. One way is to measure the  mutual information between the
  categorical label and the parameters of the categorical output. This assesses
  uncertainty in predictions due to `model uncertainty`. Model
  uncertainty can be expressed as the difference of the total uncertainty and
  the expected data uncertainty:
  `Model uncertainty = Total uncertainty - Expected data uncertainty`, where

  * `Total uncertainty`: Entropy of expected predictive distribution.
  * `Expected data uncertainty`: Expected entropy of individual predictive
    distribution.

  This formulation was given by [1, 2] and allows the decomposition of total
  uncertainty into model uncertainty and expected data uncertainty. The
  total uncertainty will be high whenever the model is uncertain. However, the
  model uncertainty, the difference between total and expected data
  uncertainty, will be non-zero iff the ensemble disagrees.

  ## References:
  [1] Depeweg, S., Hernandez-Lobato, J. M., Doshi-Velez, F, and Udluft, S.
      Decomposition of uncertainty for active learning and reliable
      reinforcement learning in stochastic systems.
      stat 1050, p.11, 2017.
  [2] Malinin, A., Mlodozeniec, B., and Gales, M.
      Ensemble Distribution Distillation.
      arXiv:1905.00076, 2019.

  Args:
    logits: Tensor, shape (N, k, nc). Logits for N instances, k ensembles and
      nc classes.

  Raises:
    TypeError: Raised if both logits and probabilities are not set or both are
      set.
    ValueError: Raised if logits or probabilities do not conform to expected
      shape.

  Returns:
    model_uncertainty: Tensor, shape (N,).
    total_uncertainty: Tensor, shape (N,).
    expected_data_uncertainty: Tensor, shape (N,).
  """

  if logits is None:
    raise TypeError(
        "model_uncertainty expected logits to be set.")
  if tf.rank(logits).numpy() != 3:
    raise ValueError(
        "model_uncertainty expected logits to be of shape (N, k, nc),"
        "instead got {}".format(logits.shape))

  # expected data uncertainty
  log_prob = tf.math.log_softmax(logits, -1)
  prob = tf.exp(log_prob)
  expected_data_uncertainty = tf.reduce_mean(
      tf.reduce_sum(- prob * log_prob, -1), -1)

  n_ens = tf.cast(log_prob.shape[1], tf.float32)
  log_expected_probabilities = tf.reduce_logsumexp(
      log_prob, 1) - tf.math.log(n_ens)
  expected_probabilities = tf.exp(log_expected_probabilities)
  total_uncertainty = tf.reduce_sum(
      - expected_probabilities * log_expected_probabilities, -1)

  model_uncertainty_ = total_uncertainty - expected_data_uncertainty

  return model_uncertainty_, total_uncertainty, expected_data_uncertainty
