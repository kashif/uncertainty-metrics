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

# Lint as: python3
"""Tests for Censored Calibration Error."""

import numpy as np
import tensorflow as tf
import uncertainty_metrics as um


class CensoredCalibrationErrorTest(tf.test.TestCase):

  def _sim(self, n):
    probs = np.random.rand(n)
    calibration_error = 0.7 * probs**2 + 0.3 * probs
    # Simulate outcomes according to this model.
    labels = (np.random.rand(n) <= calibration_error).astype(np.float)
    study_time = 1.5 - labels
    event = labels
    return (probs, study_time, event)

  def test_zero_one(self):
    n = 2000
    probs, study_time, event = self._sim(n)
    ce = um.CCE(end_time=1, folds=3, bootstrap_size=10, num_repeats=2)
    est = ce.rms_calibration_error(probs, study_time, event)
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_simple_call(self):
    n = 2000
    probs, study_time, event = self._sim(n)
    est = um.cce(
        probs,
        study_time,
        event,
        end_time=1,
        folds=3,
        bootstrap_size=10,
        num_repeats=2)
    self.assertGreaterEqual(est, 0)
    self.assertLessEqual(est, 1)

  def test_conf_int(self):
    n = 2000
    probs, study_time, event = self._sim(n)
    lower_ci, _, upper_ci = um.cce_conf_int(
        probs,
        study_time,
        event,
        end_time=1,
        folds=3,
        bootstrap_size=10,
        num_repeats=2)
    self.assertGreaterEqual(lower_ci, 0)
    self.assertLessEqual(lower_ci, 1)
    self.assertGreaterEqual(upper_ci, 0)
    self.assertLessEqual(upper_ci, 1)


if __name__ == '__main__':
  tf.test.main()
