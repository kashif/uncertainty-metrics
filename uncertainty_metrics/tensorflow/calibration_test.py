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

"""Tests for uncertainty_metrics.calibration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import uncertainty_metrics as um


class CalibrationTest(parameterized.TestCase, tf.test.TestCase):

  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

  @parameterized.parameters(
      itertools.product(_TEMPERATURES, _NLABELS, _NSAMPLES)
  )
  def test_brier_decomposition(self, temperature, nlabels, nsamples):
    """Test the accuracy of the estimated Brier decomposition."""
    tf.random.set_seed(1)
    logits = tf.random.normal((nsamples, nlabels)) / temperature
    labels = tf.random.uniform((nsamples,), maxval=nlabels, dtype=tf.int32)

    uncertainty, resolution, reliability = um.brier_decomposition(
        labels=labels, logits=logits)
    uncertainty = float(uncertainty)
    resolution = float(resolution)
    reliability = float(reliability)

    # Recover an estimate of the Brier score from the decomposition
    brier = uncertainty - resolution + reliability

    # Estimate Brier score directly
    brier_direct = um.brier_score(labels=labels, logits=logits)
    brier_direct = float(brier_direct)

    logging.info("Brier, n=%d k=%d T=%.2f, Unc %.4f - Res %.4f + Rel %.4f = "
                 "Brier %.4f,  Brier-direct %.4f",
                 nsamples, nlabels, temperature,
                 uncertainty, resolution, reliability,
                 brier, brier_direct)

    self.assertGreaterEqual(resolution, 0.0, msg="Brier resolution negative")
    self.assertGreaterEqual(reliability, 0.0, msg="Brier reliability negative")
    self.assertAlmostEqual(
        brier, brier_direct, delta=1.0e-2,
        msg="Brier from decomposition (%.4f) and Brier direct (%.4f) disagree "
        "beyond estimation error." % (brier, brier_direct))

  def _compute_perturbed_reliability(self, data, labels,
                                     weights, bias, perturbation):
    """Compute reliability of data set under perturbed hypothesis."""
    weights_perturbed = weights + perturbation*tf.random.normal(weights.shape)
    logits_perturbed = tf.matmul(data, weights_perturbed)
    logits_perturbed += tf.expand_dims(bias, 0)

    _, _, reliability = um.brier_decomposition(
        labels=labels, logits=logits_perturbed)

    return float(reliability)

  def _generate_linear_dataset(self, nfeatures, nlabels, nsamples):
    tf.random.set_seed(1)
    data = tf.random.normal((nsamples, nfeatures))
    weights = tf.random.normal((nfeatures, nlabels))
    bias = tf.random.normal((nlabels,))

    logits_true = tf.matmul(data, weights) + tf.expand_dims(bias, 0)
    prob_true = tfp.distributions.Categorical(logits=logits_true)
    labels = prob_true.sample(1)
    labels = tf.reshape(labels, (tf.size(input=labels),))

    return data, labels, weights, bias

  @parameterized.parameters(
      (5, 2, 20000), (5, 4, 20000),
  )
  def test_reliability_experiment(self, nfeatures, nlabels, nsamples,
                                  tolerance=0.05):
    data, labels, weights, bias = self._generate_linear_dataset(
        nfeatures, nlabels, nsamples)

    nreplicates = 40
    perturbations = np.linspace(0.0, 3.0, 10)
    reliability = np.zeros_like(perturbations)

    for i, perturbation in enumerate(perturbations):
      reliability_replicates = np.array(
          [self._compute_perturbed_reliability(data, labels, weights,
                                               bias, perturbation)
           for _ in range(nreplicates)])
      reliability[i] = np.mean(reliability_replicates)
      logging.info("Reliability at perturbation %.3f: %.4f", perturbation,
                   reliability[i])

    for i in range(1, len(reliability)):
      self.assertLessEqual(reliability[i-1], reliability[i] + tolerance,
                           msg="Reliability decreases (%.4f to %.4f + %.3f) "
                           "with perturbation size increasing from %.4f "
                           "to %.4f" % (reliability[i-1], reliability[i],
                                        tolerance,
                                        perturbations[i-1], perturbations[i]))

  def _generate_perfect_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and well calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()

    return logits, labels

  def _generate_random_calibration_logits(self, nsamples, nclasses):
    """Generate well distributed and poorly calibrated probabilities.

    Args:
      nsamples: int, >= 1, number of samples to generate.
      nclasses: int, >= 2, number of classes.

    Returns:
      logits: Tensor, shape (nsamples, nclasses), tf.float32, unnormalized log
        probabilities (logits) of the probabilistic predictions.
      labels: Tensor, shape (nsamples,), tf.int32, the true class labels.  Each
        element is in the range 0,..,nclasses-1.
    """
    tf.random.set_seed(1)

    logits = 2.0*tf.random.normal((nsamples, nclasses))
    py = tfp.distributions.Categorical(logits=logits)
    labels = py.sample()
    logits_other = 2.0*tf.random.normal((nsamples, nclasses))

    return logits_other, labels

  def _bayesian_ece_q(self, num_bins, logits, labels, num_ece_samples=200):
    bece_samples = um.bayesian_expected_calibration_error(
        num_bins, logits=logits, labels_true=labels,
        num_ece_samples=num_ece_samples)

    bece_q = tfp.stats.percentile(bece_samples, [1.0, 50.0, 99.0])
    bece_q1 = bece_q[0]
    bece_q50 = bece_q[1]
    bece_q99 = bece_q[2]

    return bece_q1, bece_q50, bece_q99

if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
