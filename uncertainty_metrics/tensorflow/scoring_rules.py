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
"""Scoring rules."""

import math
import tensorflow as tf
import tensorflow_probability as tfp


def brier_decomposition(labels=None, logits=None, probabilities=None):
  r"""Decompose the Brier score into uncertainty, resolution, and reliability.

  [Proper scoring rules][1] measure the quality of probabilistic predictions;
  any proper scoring rule admits a [unique decomposition][2] as
  `Score = Uncertainty - Resolution + Reliability`, where:

  * `Uncertainty`, is a generalized entropy of the average predictive
    distribution; it can both be positive or negative.
  * `Resolution`, is a generalized variance of individual predictive
    distributions; it is always non-negative.  Difference in predictions reveal
    information, that is why a larger resolution improves the predictive score.
  * `Reliability`, a measure of calibration of predictions against the true
    frequency of events.  It is always non-negative and a lower value here
    indicates better calibration.

  This method estimates the above decomposition for the case of the Brier
  scoring rule for discrete outcomes.  For this, we need to discretize the space
  of probability distributions; we choose a simple partition of the space into
  `nlabels` events: given a distribution `p` over `nlabels` outcomes, the index
  `k` for which `p_k > p_i` for all `i != k` determines the discretization
  outcome; that is, `p in M_k`, where `M_k` is the set of all distributions for
  which `p_k` is the largest value among all probabilities.

  The estimation error of each component is O(k/n), where n is the number
  of instances and k is the number of labels.  There may be an error of this
  order when compared to `brier_score`.

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
  [2]: Jochen Broecker.  Reliability, sufficiency, and the decomposition of
       proper scores.
       Quarterly Journal of the Royal Meteorological Society, Vol. 135, 2009.
       https://rmets.onlinelibrary.wiley.com/doi/epdf/10.1002/qj.456

  Args:
    labels: Tensor, (n,), with tf.int32 or tf.int64 elements containing ground
      truth class labels in the range [0,nlabels].
    logits: Tensor, (n, nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n, nlabels), with predictive probability
      distribution (alternative to logits argument).

  Returns:
    uncertainty: Tensor, scalar, the uncertainty component of the
      decomposition.
    resolution: Tensor, scalar, the resolution component of the decomposition.
    reliability: Tensor, scalar, the reliability component of the
      decomposition.
  """
  # if labels:
  #   labels = tf.convert_to_tensor(labels)
  # if logits:
  #   logits = tf.cast(logits, None)
  # if probabilities:
  #   probabilities = tf.cast(probabilities, None)
  if (logits is None) == (probabilities is None):
    raise ValueError(
        "brier_decomposition expects exactly one of logits or probabilities.")
  if probabilities is None:
    if logits.get_shape().as_list()[-1] == 1:
      raise ValueError(
          "brier_decomposition expects logits for binary classification of "
          "shape (n, 2) for nlabels=2 but got ", logits.get_shape())
    probabilities = tf.math.softmax(logits, axis=1)
  if (probabilities.get_shape().as_list()[-1] == 1 or
      len(probabilities.get_shape().as_list()) == 1):
    raise ValueError(
        "brier_decomposition expects probabilities for binary classification of"
        " shape (n, 2) for nlabels=2 but got ", probabilities.get_shape())
  _, nlabels = probabilities.shape  # Implicit rank check.

  # Compute pbar, the average distribution
  pred_class = tf.argmax(probabilities, axis=1, output_type=tf.int32)
  confusion_matrix = tf.math.confusion_matrix(pred_class, labels, nlabels,
                                              dtype=tf.float32)
  dist_weights = tf.reduce_sum(confusion_matrix, axis=1)
  dist_weights /= tf.reduce_sum(dist_weights)
  pbar = tf.reduce_sum(confusion_matrix, axis=0)
  pbar /= tf.reduce_sum(pbar)

  # dist_mean[k,:] contains the empirical distribution for the set M_k
  # Some outcomes may not realize, corresponding to dist_weights[k] = 0
  dist_mean = confusion_matrix / tf.expand_dims(
      tf.reduce_sum(confusion_matrix, axis=1) + 1.0e-7, 1)

  # Uncertainty: quadratic entropy of the average label distribution
  uncertainty = -tf.reduce_sum(tf.square(pbar))

  # Resolution: expected quadratic divergence of predictive to mean
  resolution = tf.square(tf.expand_dims(pbar, 1) - dist_mean)
  resolution = tf.reduce_sum(dist_weights * tf.reduce_sum(resolution, axis=1))

  # Reliability: expected quadratic divergence of predictive to true
  prob_true = tf.gather(dist_mean, pred_class, axis=0)
  reliability = tf.reduce_sum(tf.square(prob_true - probabilities), axis=1)
  reliability = tf.reduce_mean(reliability)

  return uncertainty, resolution, reliability


def brier_score(labels=None, logits=None, probabilities=None, aggregate=True):
  r"""Compute the Brier score for a probabilistic prediction.

  The [Brier score][1] is a loss function for probabilistic predictions over a
  number of discrete outcomes.  For a probability vector `p` and a realized
  outcome `k` the Brier score is `sum_i p[i]*p[i] - 2*p[k]`.  Smaller values are
  better in terms of prediction quality.  The Brier score can be negative.

  Compared to the cross entropy (aka logarithmic scoring rule) the Brier score
  does not strongly penalize events which are deemed unlikely but do occur,
  see [2].  The Brier score is a strictly proper scoring rule and therefore
  yields consistent probabilistic predictions.

  #### References
  [1]: G.W. Brier.
       Verification of forecasts expressed in terms of probability.
       Monthley Weather Review, 1950.
  [2]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

  Args:
    labels: Tensor, (n,), with tf.int32 or tf.int64 elements containing ground
      truth class labels in the range [0,nlabels].
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n, nlabels), with predictive probability
      distribution (alternative to logits argument).
    aggregate: bool, whether or not to average over the batch.

  Returns:
    brier_score: Tensor, if `aggregate` is true then it is a scalar, the average
      Brier score over all instances, else it is a vector of the Brier score for
      each individual element of the batc.
  """
  if (logits is None) == (probabilities is None):
    raise ValueError(
        "brier_score expects exactly one of logits or probabilities.")
  if probabilities is None:
    if logits.get_shape().as_list()[-1] == 1:
      raise ValueError(
          "brier_score expects logits for binary classification of "
          "shape (n, 2) for nlabels=2 but got ", logits.get_shape())
    probabilities = tf.math.softmax(logits, axis=1)
  if (probabilities.get_shape().as_list()[-1] == 1 or
      len(probabilities.get_shape().as_list()) == 1):
    raise ValueError(
        "brier_score expects probabilities for binary classification of"
        " shape (n, 2) for nlabels=2 but got ", probabilities.get_shape())
  _, nlabels = probabilities.shape  # Implicit rank check.
  plabel = tf.reduce_sum(tf.one_hot(labels, nlabels) * probabilities, axis=1)

  brier = tf.reduce_sum(tf.square(probabilities), axis=1) - 2.0 * plabel
  if aggregate:
    brier = tf.reduce_mean(brier)

  return brier


def crps_score(labels=None, predictive_samples=None):
  r"""Computes the Continuous Ranked Probability Score (CRPS).

  The Continuous Ranked Probability Score is a [proper scoring rule][1] for
  assessing the probabilistic predictions of a model against a realized value.
  The CRPS is

  \\(\textrm{CRPS}(F,y) = \int_{-\inf}^{\inf} (F(z) - 1_{z \geq y})^2 dz.\\)

  Here \\(F\\) is the cumulative distribution function of the model predictive
  distribution and \\(y)\\ is the realized ground truth value.

  The CRPS can be used as a loss function for training an implicit model for
  probabilistic regression.  It can also be used to assess the predictive
  performance of a probabilistic regression model.

  In this implementation we use an equivalent representation of the CRPS,

  \\(\textrm{CRPS}(F,y) = E_{z~F}[|z-y|] - (1/2) E_{z,z'~F}[|z-z'|].\\)

  This equivalent representation has an unbiased sample estimate and our
  implementation of the CRPS has a complexity is O(n m).

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

  Args:
    labels: Tensor, (n,), with tf.float32 or tf.float64 real-valued targets.
    predictive_samples: Tensor, (n,m), with tf.float32 or tf.float64 values.
      Each row at [i,:] contains m samples of the model predictive
      distribution, p(y|x_i).

  Returns:
    crps: (n,) Tensor, the CRPS score for each instance; a lower score
      indicates a better fit.
  """
  if labels is None:
    raise ValueError("target labels must be provided")
  if labels.shape.ndims != 1:
    raise ValueError("target labels must be of rank 1")

  if predictive_samples is None:
    raise ValueError("predictive samples must be provided")
  if predictive_samples.shape.ndims != 2:
    raise ValueError("predictive samples must be of rank 2")
  if predictive_samples.shape[0] != labels.shape[0]:
    raise ValueError("first dimension of predictive samples shape "
                     "must match target labels shape")

  pairwise_diff = tf.roll(predictive_samples, 1, axis=1) - predictive_samples
  predictive_diff = tf.abs(pairwise_diff)
  estimated_dist_pairwise = tf.reduce_mean(input_tensor=predictive_diff, axis=1)

  labels = tf.expand_dims(labels, 1)
  dist_realization = tf.reduce_mean(tf.abs(predictive_samples-labels), axis=1)

  crps = dist_realization - 0.5*estimated_dist_pairwise

  return crps


def crps_normal_score(labels=None, means=None, stddevs=None):
  r"""Computes the CRPS against a Normal predictive distribution.

  For a predictive Normal distribution the CRPS has a
  [closed-form solution][1].  Our implementation has a complexity O(n).

  #### References
  [1]: Tilmann Gneiting, Adrian E. Raftery.
       Strictly Proper Scoring Rules, Prediction, and Estimation.
       Journal of the American Statistical Association, Vol. 102, 2007.
       https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

  Args:
    labels: Tensor, (n,), with tf.float32 or tf.float64 real-valued targets.
    means: Tensor, (n,), with tf.float32 or tf.float64 real-valued preditive
      means.
    stddevs: Tensor, (n,), with tf.float32 or tf.float64 real-valued preditive
      standard deviations.

  Returns:
    crps: (n,) Tensor, the CRPS score for each instance; a lower score indicates
      a better fit.
  """
  if labels is None:
    raise ValueError("target labels must be provided")
  if labels.shape.ndims != 1:
    raise ValueError("target labels must be of rank 1")

  if means is None:
    raise ValueError("predictive means must be provided")
  if means.shape != labels.shape:
    raise ValueError("predictive means shape must match target labels shape")
  if stddevs is None:
    raise ValueError("predictive standard deviations must be provided")
  if stddevs.shape != labels.shape:
    raise ValueError("predictive standard deviations shape "
                     "must match target labels shape")

  dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
  labels_std = (labels - means) / stddevs

  crps = 2.0*dist.prob(labels_std) + labels_std*(2.0*dist.cdf(labels_std)-1.0)
  crps -= 1.0/math.sqrt(math.pi)
  crps *= stddevs

  return crps
