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

"""Tests for Oracle-Model Collaborative Accuracy."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import uncertainty_metrics as um


class OracleCollaborativeAccuracyTest(parameterized.TestCase, tf.test.TestCase):

  def testOracleCollaborativeAccuracy(self):
    num_bins = 10
    fraction = 0.4
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array(
        [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71, 0.81 + 0.85, 0])
    # `(3 - 1)` refers to the rest examples in this bin
    # (minus the examples sent to the moderators), while `2/3` is
    # the accuracy in this bin.
    bin_collab_correct_sums = np.array(
        [0, 0, 0, 0, 0, 2, 1 * 1.0 + (3 - 1) * (2 / 3), 0, 2, 0])

    correct_acc = np.sum(bin_collab_correct_sums) / np.sum(bin_counts)

    metric = um.OracleCollaborativeAccuracy(
        fraction, num_bins, name='collab_acc', dtype=tf.float64)

    acc1 = metric(labels, pred_probs)
    self.assertAllClose(acc1, correct_acc)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    actual_bin_bin_collab_correct_sums = tf.convert_to_tensor(
        metric.collab_correct_sums)

    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)
    self.assertAllClose(bin_collab_correct_sums,
                        actual_bin_bin_collab_correct_sums)


if __name__ == '__main__':
  tf.test.main()
