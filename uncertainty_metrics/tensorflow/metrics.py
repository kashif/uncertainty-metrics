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
"""Calibration metrics."""
import tensorflow.compat.v2 as tf


class ExpectedCalibrationError(tf.keras.metrics.Metric):
  """Expected Calibration Error.

  Expected calibration error (Guo et al., 2017, Naeini et al., 2015) is a scalar
  measure of calibration for probabilistic models. Calibration is defined as the
  level to which the accuracy over a set of predicted decisions and true
  outcomes associated with a given predicted probability level matches the
  predicted probability. A perfectly calibrated model would be correct `p`% of
  the time for all examples for which the predicted probability was `p`%, over
  all values of `p`.

  This metric can be computed as follows. First, cut up the probability space
  interval [0, 1] into some number of bins. Then, for each example, store the
  predicted class (based on a threshold of 0.5 in the binary case and the max
  probability in the multiclass case), the predicted probability corresponding
  to the predicted class, and the true label into the corresponding bin based on
  the predicted probability. Then, for each bin, compute the average predicted
  probability ("confidence"), the accuracy of the predicted classes, and the
  absolute difference between the confidence and the accuracy ("calibration
  error"). Expected calibration error can then be computed as a weighted average
  calibration error over all bins, weighted based on the number of examples per
  bin.

  Perfect calibration under this setup is when, for all bins, the average
  predicted probability matches the accuracy, and thus the expected calibration
  error equals zero. In the limit as the number of bins goes to infinity, the
  predicted probability would be equal to the accuracy for all possible
  probabilities.

  References:
    1. Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. On Calibration of Modern
       Neural Networks. in International Conference on Machine Learning (ICML)
       cs.LG, (Cornell University Library, 2017).
    2. Naeini, M. P., Cooper, G. F. & Hauskrecht, M. Obtaining Well Calibrated
       Probabilities Using Bayesian Binning. Proc Conf AAAI Artif Intell 2015,
       2901-2907 (2015).
  """

  _setattr_tracking = False  # Automatic tracking breaks some unit tests

  def __init__(self, num_bins=15, name=None, dtype=None):
    """Constructs an expected calibration error metric.

    Args:
      num_bins: Number of bins to maintain over the interval [0, 1].
      name: Name of this metric.
      dtype: Data type.
    """
    super(ExpectedCalibrationError, self).__init__(name, dtype)
    self.num_bins = num_bins

    self.correct_sums = self.add_weight(
        'correct_sums', shape=(num_bins,), initializer=tf.zeros_initializer)
    self.prob_sums = self.add_weight(
        'prob_sums', shape=(num_bins,), initializer=tf.zeros_initializer)
    self.counts = self.add_weight(
        'counts', shape=(num_bins,), initializer=tf.zeros_initializer)

  @tf.function
  def update_state(self, labels, probabilities, **kwargs):
    """Updates this metric.

    This will flatten the labels and probabilities, and then compute the ECE
    over all predictions.

    Args:
      labels: Tensor of shape [..., ] of class labels in [0, k-1].
      probabilities: Tensor of shape [..., ], [..., 1] or [..., k] of normalized
        probabilities associated with the True class in the binary case, or with
        each of k classes in the multiclass case.
      **kwargs: Other potential keywords, which will be ignored by this method.
    """
    del kwargs  # unused
    labels = tf.convert_to_tensor(labels)
    probabilities = tf.cast(probabilities, self.dtype)

    # Flatten labels to [N, ] and probabilities to [N, 1] or [N, k].
    if tf.rank(labels) != 1:
      labels = tf.reshape(labels, [-1])
    if tf.rank(probabilities) != 2 or (tf.shape(probabilities)[0] !=
                                       tf.shape(labels)[0]):
      probabilities = tf.reshape(probabilities, [tf.shape(labels)[0], -1])
    # Extend any probabilities of shape [N, 1] to shape [N, 2].
    # NOTE: XLA does not allow for different shapes in the branches of a
    # conditional statement. Therefore, explicit indexing is used.
    given_k = tf.shape(probabilities)[-1]
    k = tf.math.maximum(2, given_k)
    probabilities = tf.cond(
        given_k < 2,
        lambda: tf.concat([1. - probabilities, probabilities], axis=-1)[:, -k:],
        lambda: probabilities)

    pred_labels = tf.math.argmax(probabilities, axis=-1)
    pred_probs = tf.math.reduce_max(probabilities, axis=-1)
    correct_preds = tf.math.equal(pred_labels,
                                  tf.cast(labels, pred_labels.dtype))
    correct_preds = tf.cast(correct_preds, self.dtype)

    bin_indices = tf.histogram_fixed_width_bins(
        pred_probs, tf.constant([0., 1.], self.dtype), nbins=self.num_bins)
    batch_correct_sums = tf.math.unsorted_segment_sum(
        data=tf.cast(correct_preds, self.dtype),
        segment_ids=bin_indices,
        num_segments=self.num_bins)
    batch_prob_sums = tf.math.unsorted_segment_sum(data=pred_probs,
                                                   segment_ids=bin_indices,
                                                   num_segments=self.num_bins)
    batch_counts = tf.math.unsorted_segment_sum(data=tf.ones_like(bin_indices),
                                                segment_ids=bin_indices,
                                                num_segments=self.num_bins)
    batch_counts = tf.cast(batch_counts, self.dtype)
    self.correct_sums.assign_add(batch_correct_sums)
    self.prob_sums.assign_add(batch_prob_sums)
    self.counts.assign_add(batch_counts)

  def result(self):
    """Computes the expected calibration error."""
    non_empty = tf.math.not_equal(self.counts, 0)
    correct_sums = tf.boolean_mask(self.correct_sums, non_empty)
    prob_sums = tf.boolean_mask(self.prob_sums, non_empty)
    counts = tf.boolean_mask(self.counts, non_empty)
    accs = correct_sums / counts
    confs = prob_sums / counts
    total_count = tf.reduce_sum(counts)
    return tf.reduce_sum(counts / total_count * tf.abs(accs - confs))

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    tf.keras.backend.batch_set_value([(v, [0.,]*self.num_bins) for v in
                                      self.variables])
