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
"""Oracle Collaborative Accuracy measures for probabilistic predictions.

Oracle Collaborative Accuracy measures the usefulness of model uncertainty
scores in facilitating human-computer collaboration (e.g., between a neural
model and an "oracle" human moderator in moderating online toxic comments).

The idea is that given a large amount of testing examples, the model will first
generate predictions for all examples, and then send a certain percentage of
examples that it is not confident about to the human moderators, whom we assume
can label those examples correctly.

The goal of this metric is to understand, under capacity constraints on the
human moderator (e.g., the model is only allowed to send 0.1% of total examples
to humans), how well the model can collaborate with the human to achieve the
highest overall accuracy. In this way, the metric attempts to quantify the
behavior of the full model-moderator system rather than of the model alone.

A model that collaborates with a human oracle well should not be accurate, but
also capable of quantifying its uncertainty well (i.e., its uncertainty should
be calibrated such that uncertainty ≅ model accuracy).
"""
import tensorflow as tf
from uncertainty_metrics.tensorflow import calibration


def _bin_probabilities(num_bins, index, dtype):
  """Computing corresponding probabilities.

  Args:
    num_bins: Number of bins to maintain over the interval [0, 1], and the bins
      are uniformly spaced.
    index: Which index to return.
    dtype: Data type

  Returns:
    bin_probabilities: Tensor, the corresponding probabilities.
  """
  return tf.cast(tf.linspace(0.0 + 1.0 / num_bins, 1.0, num_bins)[index], dtype)


class OracleCollaborativeAccuracy(calibration.ExpectedCalibrationError):
  """Oracle Collaborative Accuracy."""

  def __init__(self, fraction=0.01, num_bins=100, name=None, dtype=None):
    """Constructs an expected collaborative accuracy metric.

    Args:
      fraction: the fraction of total examples to send to moderators.
      num_bins: Number of bins to maintain over the interval [0, 1].
      name: Name of this metric.
      dtype: Data type.
    """
    super(OracleCollaborativeAccuracy, self).__init__(num_bins, name, dtype)
    self.fraction = fraction
    self.collab_correct_sums = self.add_weight(
        "collab_correct_sums",
        shape=(num_bins,),
        initializer=tf.zeros_initializer)

  def result(self):
    """Computes the expected calibration error."""
    num_total_example = tf.reduce_sum(self.counts)
    num_oracle_examples = tf.cast(
        int(num_total_example * self.fraction), self.dtype)
    # TODO(lzi): compute the expected number of accurate predictions
    collab_correct_sums = []
    num_oracle_examples_so_far = 0.0
    for i in range(self.num_bins):
      cur_bin_counts = self.counts[i]
      cur_bin_num_correct = self.correct_sums[i]
      if num_oracle_examples_so_far + cur_bin_counts <= num_oracle_examples:
        # Send all examples in the current bin to the oracle.
        cur_bin_num_correct = cur_bin_counts
        num_oracle_examples_so_far += cur_bin_num_correct
      elif num_oracle_examples_so_far < num_oracle_examples:
        # Send num_correct_oracle examples in the current bin to oracle,
        # and have model to predict the rest.
        cur_bin_accuracy = cur_bin_num_correct / cur_bin_counts
        num_correct_oracle = tf.cast(
            num_oracle_examples - num_oracle_examples_so_far, self.dtype)
        num_correct_model = (cur_bin_counts -
                             num_correct_oracle) * cur_bin_accuracy
        cur_bin_num_correct = num_correct_oracle + num_correct_model
        num_oracle_examples_so_far = num_oracle_examples

      collab_correct_sums.append(cur_bin_num_correct)

    self.collab_correct_sums = tf.stack(collab_correct_sums)

    non_empty = tf.math.not_equal(self.counts, 0)
    counts = tf.boolean_mask(self.counts, non_empty)
    collab_correct_sums = tf.boolean_mask(self.collab_correct_sums, non_empty)

    return tf.reduce_sum(collab_correct_sums) / tf.reduce_sum(counts)
