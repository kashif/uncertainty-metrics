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

"""Tests for predictive_metrics.regression."""

import math
from absl import logging
import tensorflow as tf

from uncertainty_metrics import regression


class RegressionTest(tf.test.TestCase):

  def test_normal_predictive_agreement_analytic_vs_sampling_approx(self):
    """Check that analytic CRPS and sample approximation CRPS agree.
    """
    tf.random.set_seed(1)

    nsamples = 100
    npredictive_samples = 10000

    labels = tf.random.normal((nsamples,))
    predictive_samples = tf.random.normal((nsamples, npredictive_samples))
    crps_sample = regression.crps_score(
        labels=labels, predictive_samples=predictive_samples)

    means = tf.zeros_like(labels)
    stddevs = tf.ones_like(labels)
    crps_analytic = regression.crps_normal_score(labels=labels,
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
      crps_sample = regression.crps_score(
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
