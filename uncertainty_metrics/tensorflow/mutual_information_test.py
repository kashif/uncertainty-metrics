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

"""Tests for predictive_metrics.mutual_information."""

import tensorflow.compat.v1 as tf  # tf
from uncertainty_metrics import mutual_information as mi


eps = 1e-6


class MutualInformationTest(tf.test.TestCase):

  def setUp(self, seed=42):
    super(MutualInformationTest, self).setUp()
    tf.set_random_seed(seed)
    self.logits = tf.random_normal((128, 32, 10))
    prob = tf.nn.softmax(self.logits)

    prob = tf.clip_by_value(prob, eps, 1 - eps)
    expected_data_uncertainty = tf.reduce_mean(
        tf.reduce_sum(- prob * tf.math.log(prob), -1), -1)

    expected_prob = tf.reduce_mean(prob, 1)
    total_uncertainty = tf.reduce_sum(
        - expected_prob * tf.math.log(expected_prob), -1)

    model_uncertainty = total_uncertainty - expected_data_uncertainty

    self.model_uncertainty = model_uncertainty
    self.total_uncertainty = total_uncertainty
    self.expected_data_uncertainty = expected_data_uncertainty

  def assertTensorsAlmostEqual(self, x, y, msg=None):
    abs_delta = tf.abs(tf.cast(x - y, tf.float32))
    msg += "(max delta: {:.4E})".format(tf.reduce_max(abs_delta).numpy())
    self.assertTrue(
        tf.reduce_all(abs_delta < eps), msg=msg)

  def test_model_uncertainty_with_logits(self):
    output, _, _ = mi.model_uncertainty(logits=self.logits)
    msg = "Model uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(self.model_uncertainty, output, msg=msg)

  def test_total_uncertainty_with_logits(self):
    _, output, _ = mi.model_uncertainty(logits=self.logits)
    msg = "Total uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(self.total_uncertainty, output, msg=msg)

  def test_expected_data_uncertainty_with_logits(self):
    _, _, output = mi.model_uncertainty(logits=self.logits)
    msg = "Data uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(
        self.expected_data_uncertainty, output, msg=msg)

  def test_wrong_logits_shape(self):
    logits_wrong = tf.reduce_mean(self.logits, -1)
    with self.assertRaises(ValueError):
      mi.model_uncertainty(logits=logits_wrong)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
