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

"""Metrics for model diversity."""

import itertools
import tensorflow as tf


def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.reduce_mean(tf.cast(preds_1 != preds_2, tf.float32))


def double_fault(logits_1, logits_2, labels):
  """Double fault [1] is the number of examples both classifiers predict wrong.

  Args:
    logits_1: tf.Tensor.
    logits_2: tf.Tensor.
    labels: tf.Tensor.

  Returns:
    Scalar double-fault diversity metric.

  ## References

  [1] Kuncheva, Ludmila I., and Christopher J. Whitaker. "Measures of diversity
      in classifier ensembles and their relationship with the ensemble
      accuracy." Machine learning 51.2 (2003): 181-207.
  """
  preds_1 = tf.cast(tf.argmax(logits_1, axis=-1), labels.dtype)
  preds_2 = tf.cast(tf.argmax(logits_2, axis=-1), labels.dtype)

  fault_1_idx = tf.squeeze(tf.where(preds_1 != labels))
  fault_1_idx = tf.cast(fault_1_idx, tf.int32)

  preds_2_at_idx = tf.gather(preds_2, fault_1_idx)
  labels_at_idx = tf.gather(labels, fault_1_idx)

  double_faults = preds_2_at_idx != labels_at_idx
  double_faults = tf.cast(double_faults, tf.float32)
  return tf.reduce_mean(double_faults)


def logit_kl_divergence(logits_1, logits_2):
  """Average KL divergence between logit distributions of two classifiers."""
  probs_1 = tf.nn.softmax(logits_1)
  probs_2 = tf.nn.softmax(logits_2)
  vals = kl_divergence(probs_1, probs_2)
  return tf.reduce_mean(vals)


def kl_divergence(p, q, clip=False):
  """Generalized KL divergence [1] for unnormalized distributions.

  Args:
    p: tf.Tensor.
    q: tf.Tensor.
    clip: bool.

  Returns:
    tf.Tensor of the Kullback-Leibler divergences per example.

  ## References

  [1] Lee, Daniel D., and H. Sebastian Seung. "Algorithms for non-negative
  matrix factorization." Advances in neural information processing systems.
  2001.
  """
  if clip:
    p = tf.clip_by_value(p, tf.keras.backend.epsilon(), 1)
    q = tf.clip_by_value(q, tf.keras.backend.epsilon(), 1)
  return tf.reduce_sum(p * tf.math.log(p / q), axis=-1)


def lp_distance(x, y, p=1):
  """l_p distance."""
  diffs_abs = tf.abs(x - y)
  summation = tf.reduce_sum(tf.math.pow(diffs_abs, p), axis=-1)
  return tf.reduce_mean(tf.math.pow(summation, 1./p), axis=-1)


def cosine_distance(x, y):
  """Cosine distance between vectors x and y."""
  x_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=-1))
  x_norm = tf.reshape(x_norm, (-1, 1))
  y_norm = tf.math.sqrt(tf.reduce_sum(tf.pow(y, 2), axis=-1))
  y_norm = tf.reshape(y_norm, (-1, 1))
  normalized_x = x / x_norm
  normalized_y = y / y_norm
  return tf.reduce_mean(tf.reduce_sum(normalized_x * normalized_y, axis=-1))


# TODO(ghassen): we could extend this to take an arbitrary list of metric fns.
def average_pairwise_diversity(probs, num_models, error=None):
  """Average pairwise distance computation across models."""
  if probs.shape[0] != num_models:
    raise ValueError('The number of models {0} does not match '
                     'the probs length {1}'.format(num_models, probs.shape[0]))

  pairwise_disagreement = []
  pairwise_kl_divergence = []
  pairwise_cosine_distance = []
  for pair in list(itertools.combinations(range(num_models), 2)):
    probs_1 = probs[pair[0]]
    probs_2 = probs[pair[1]]
    pairwise_disagreement.append(disagreement(probs_1, probs_2))
    pairwise_kl_divergence.append(
        tf.reduce_mean(kl_divergence(probs_1, probs_2)))
    pairwise_cosine_distance.append(cosine_distance(probs_1, probs_2))

  # TODO(ghassen): we could also return max and min pairwise metrics.
  average_disagreement = tf.reduce_mean(tf.stack(pairwise_disagreement))
  if error is not None:
    average_disagreement /= (error + tf.keras.backend.epsilon())
  average_kl_divergence = tf.reduce_mean(tf.stack(pairwise_kl_divergence))
  average_cosine_distance = tf.reduce_mean(tf.stack(pairwise_cosine_distance))

  return {
      'disagreement': average_disagreement,
      'average_kl': average_kl_divergence,
      'cosine_similarity': average_cosine_distance
  }


def variance_bound(probs, labels, num_models):
  """Empirical upper bound on the variance for an ensemble model.

  This term was introduced in arxiv.org/abs/1912.08335 to obtain a tighter
  PAC-Bayes upper bound; we use the empirical variance of Theorem 4.

  Args:
    probs: tensor of shape `[num_models, batch_size, n_classes]`.
    labels: tensor of sparse labels, of shape `[batch_size]`.
    num_models: number of models in the ensemble.

  Returns:
    A (float) upper bound on the empirical ensemble variance.
  """
  if probs.shape[0] != num_models:
    raise ValueError('The number of models {0} does not match '
                     'the probs length {1}'.format(num_models, probs.shape[0]))
  batch_size = probs.shape[1]
  labels = tf.cast(labels, dtype=tf.int32)

  # batch_indices maps point `i` to its associated label `l_i`.
  batch_indices = tf.stack([tf.range(batch_size), labels], axis=1)
  # Shape: [num_models, batch_size, batch_size].
  batch_indices = batch_indices * tf.ones([num_models, 1, 1], dtype=tf.int32)

  # Replicate batch_indices across the `num_models` index.
  ensemble_indices = tf.reshape(tf.range(num_models), [num_models, 1, 1])
  ensemble_indices = ensemble_indices * tf.ones([1, batch_size, 1],
                                                dtype=tf.int32)
  # Shape: [num_models, batch_size, n_classes].
  indices = tf.concat([ensemble_indices, batch_indices], axis=-1)

  # Shape: [n_models, n_points].
  # per_model_probs[n, i] contains the probability according to model `n` that
  # point `i` in the batch has its true label.
  per_model_probs = tf.gather_nd(probs, indices)

  max_probs = tf.reduce_max(per_model_probs, axis=0)  # Shape: [n_points]
  avg_probs = tf.reduce_mean(per_model_probs, axis=0)  # Shape: [n_points]

  return .5 * tf.reduce_mean(
      tf.square((per_model_probs - avg_probs) / max_probs))
