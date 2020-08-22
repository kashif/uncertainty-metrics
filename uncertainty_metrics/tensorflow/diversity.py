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


# TODO(ghassen): disagreement and double_fault could be extended beyond pairs.
def disagreement(logits_1, logits_2):
  """Disagreement between the predictions of two classifiers."""
  preds_1 = tf.argmax(logits_1, axis=-1, output_type=tf.int32)
  preds_2 = tf.argmax(logits_2, axis=-1, output_type=tf.int32)
  return tf.reduce_mean(tf.cast(preds_1 != preds_2, tf.float32))


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
