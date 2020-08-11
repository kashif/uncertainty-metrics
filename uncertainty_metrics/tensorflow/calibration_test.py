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

"""Tests for calibration."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import uncertainty_metrics as um


class CalibrationTest(parameterized.TestCase, tf.test.TestCase):

  _TEMPERATURES = [0.01, 1.0, 5.0]
  _NLABELS = [2, 4]
  _NSAMPLES = [8192, 16384]

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

  def testECEBinaryClassification(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = np.array([0, 0, 0, 0, 0, 2, 3, 1, 2, 0])
    bin_correct_sums = np.array([0, 0, 0, 0, 0, 1, 2, 0, 2, 0])
    bin_prob_sums = np.array([0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68,
                              0.71, 0.81 + 0.85, 0])

    correct_ece = 0.
    bin_accs = np.array([0.] * num_bins)
    bin_confs = np.array([0.] * num_bins)
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = um.ExpectedCalibrationError(
        num_bins, name='ECE', dtype=tf.float64)
    self.assertLen(metric.variables, 3)

    ece1 = metric(labels, pred_probs)
    self.assertAllClose(ece1, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

    # Test various types of input shapes.
    metric.reset_states()
    metric.update_state(labels[:2], pred_probs[:2])
    metric.update_state(labels[2:6].reshape(2, 2),
                        pred_probs[2:6].reshape(2, 2))
    metric.update_state(labels[6:7], pred_probs[6:7])
    ece2 = metric(labels[7:, np.newaxis], pred_probs[7:, np.newaxis])
    ece3 = metric.result()
    self.assertAllClose(ece2, ece3)
    self.assertAllClose(ece3, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def testECEBinaryClassificationKerasModel(self):
    num_bins = 10
    pred_probs = np.array([0.51, 0.45, 0.39, 0.66, 0.68, 0.29, 0.81, 0.85])
    # max_pred_probs: [0.51, 0.55, 0.61, 0.66, 0.68, 0.71, 0.81, 0.85]
    # pred_class: [1, 0, 0, 1, 1, 0, 1, 1]
    labels = np.array([0., 0., 0., 1., 0., 1., 1., 1.])
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 0, 0, 2, 3, 1, 2, 0]
    bin_correct_sums = [0, 0, 0, 0, 0, 1, 2, 0, 2, 0]
    bin_prob_sums = [0, 0, 0, 0, 0, 0.51 + 0.55, 0.61 + 0.66 + 0.68, 0.71,
                     0.81 + 0.85, 0]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = um.ExpectedCalibrationError(num_bins, name='ECE')
    self.assertLen(metric.variables, 3)

    model = tf.keras.models.Sequential([tf.keras.layers.Lambda(lambda x: 1*x)])
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=[metric])
    outputs = model.predict(pred_probs)
    self.assertAllClose(pred_probs.reshape([n, 1]), outputs)
    _, ece = model.evaluate(pred_probs, labels)
    self.assertAllClose(ece, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

  def testECEMulticlassClassification(self):
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 3, 0, 0, 2, 0, 0, 1]
    bin_correct_sums = [0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0, 0, 0, 0.32 + 0.37 + 0.39, 0, 0, 0.61 + 0.65, 0, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    metric = um.ExpectedCalibrationError(
        num_bins, name='ECE', dtype=tf.float64)
    self.assertLen(metric.variables, 3)

    metric.update_state(labels[:4], pred_probs[:4])
    ece1 = metric(labels[4:], pred_probs[4:])
    ece2 = metric.result()
    self.assertAllClose(ece1, ece2)
    self.assertAllClose(ece2, correct_ece)

    actual_bin_counts = tf.convert_to_tensor(metric.counts)
    actual_bin_correct_sums = tf.convert_to_tensor(metric.correct_sums)
    actual_bin_prob_sums = tf.convert_to_tensor(metric.prob_sums)
    self.assertAllEqual(bin_counts, actual_bin_counts)
    self.assertAllEqual(bin_correct_sums, actual_bin_correct_sums)
    self.assertAllClose(bin_prob_sums, actual_bin_prob_sums)

if __name__ == '__main__':
  tf.test.main()
