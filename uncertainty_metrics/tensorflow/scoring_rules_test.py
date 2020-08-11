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

"""Tests for scoring rules."""

import itertools
import math
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import uncertainty_metrics as um


class ScoringRulesTest(parameterized.TestCase, tf.test.TestCase):

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

  def test_normal_predictive_agreement_analytic_vs_sampling_approx(self):
    """Check that analytic CRPS and sample approximation CRPS agree.
    """
    tf.random.set_seed(1)

    nsamples = 100
    npredictive_samples = 10000

    labels = tf.random.normal((nsamples,))
    predictive_samples = tf.random.normal((nsamples, npredictive_samples))
    crps_sample = um.crps_score(
        labels=labels, predictive_samples=predictive_samples)

    means = tf.zeros_like(labels)
    stddevs = tf.ones_like(labels)
    crps_analytic = um.crps_normal_score(labels=labels,
                                         means=means,
                                         stddevs=stddevs)

    max_diff = tf.reduce_max(tf.abs(crps_sample - crps_analytic))
    max_diff = float(max_diff)

    # CRPS is at most 1, so tolerance is an upper bound to 5*SEM
    tolerance = 5.0 / math.sqrt(npredictive_samples)
    logging.info("Maximum difference %.4f, allowed tolerance %.4f",
                 max_diff, tolerance)

    self.assertLessEqual(max_diff, tolerance,
                         msg="Sample-CRPS differs from analytic-CRPS "
                         "by %.4f > %.4f" % (max_diff, tolerance))

  def test_crps_increases_with_increasing_deviation_in_mean(self):
    """Assert that the CRPS score increases when we increase the mean.
    """
    tf.random.set_seed(1)

    nspacing = 10
    npredictive_samples = 10000
    ntrue_samples = 1000

    # (nspacing,npredictive_samples) samples from N(mu_i, 1)
    predictive_samples = tf.random.normal((nspacing, npredictive_samples))
    predictive_samples += tf.expand_dims(tf.linspace(0.0, 5.0, nspacing), 1)

    crps_samples = []
    for _ in range(ntrue_samples):
      labels = tf.random.normal((nspacing,))
      crps_sample = um.crps_score(
          labels=labels, predictive_samples=predictive_samples)
      crps_samples.append(crps_sample)

    crps_samples = tf.stack(crps_samples, 1)
    crps_average = tf.reduce_mean(crps_samples, axis=1)
    crps_average = crps_average.numpy()

    # The average should be monotonically increasing
    for i in range(1, len(crps_average)):
      crps_cur = crps_average[i]
      crps_prev = crps_average[i-1]
      logging.info("CRPS cur %.5f, prev %.5f", crps_cur, crps_prev)
      self.assertLessEqual(crps_prev, crps_cur,
                           msg="CRPS violates monotonicity in mean")


if __name__ == "__main__":
  tf.test.main()
