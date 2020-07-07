# coding=utf-8
# Copyright 2020 The uncertainty_metrics Authors.
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
"""Calibration metrics for probabilistic predictions.

Calibration is a property of probabilistic prediction models: a model is said to
be well-calibrated if its predicted probabilities over a class of events match
long-term frequencies over the sampling distribution.
"""
import numpy as np

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


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


def _bin_centers(num_bins):
  """Split the unit interval into `num_bins` equal parts and return centers."""
  return tf.linspace(0.0, 1.0 - 1.0/num_bins, num_bins) + 0.5/num_bins


def _prob_posterior(num_bins, bin_n, bin_pmean):
  """Compute posterior mean and standard deviation for unknown per-bin means.

  Args:
    num_bins: int, positive, the number of equal-sized bins to divide the unit
      interval into.
    bin_n: Tensor, shape (num_bins,), containing the absolute counts of
      observations falling into each bin.
    bin_pmean: Tensor, shape (num_bins,), containing the average predicted
      probability value that falls into this bin.

  Returns:
    posterior_means: Tensor, (num_bins,), the per-bin posterior mean.
    posterior_stddev: Tensor, (num_bins,), the per-bin posterior standard
      deviation.
  """
  bin_centers = _bin_centers(num_bins)
  sigma0_sq = (1.0/num_bins) / 12.0   # matching moment of Uniform(Bin)
  sigma_sq = (1.0/num_bins) / 12.0

  sigman_sq = 1.0 / (bin_n/sigma_sq + 1.0/sigma0_sq)
  posterior_means = sigman_sq*(
      bin_centers/sigma0_sq + bin_n*(bin_pmean/sigma_sq))
  posterior_stddev = tf.sqrt(sigman_sq)

  return posterior_means, posterior_stddev


def _compute_ece(prob, bin_mean_prob):
  """Compute the expected calibration error (ECE).

  Args:
    prob: Tensor, shape (2,num_bins), containing the probabilities over the
      {incorrect,correct}x{0,1,..,num_bins-1} events.
    bin_mean_prob: Tensor, shape (1,num_bins), containing the average
      probability within each bin.

  Returns:
    ece: Tensor, scalar, the expected calibration error.
  """
  pz_given_b = prob / tf.expand_dims(tf.reduce_sum(prob, axis=0), 0)
  prob_correct = prob[1, :] / tf.reduce_sum(prob[1, :])
  ece = tf.reduce_sum(prob_correct * tf.abs(pz_given_b[1, :] - bin_mean_prob))

  return ece


def _compute_calibration_bin_statistics(
    num_bins, logits=None, probabilities=None,
    labels_true=None, labels_predicted=None):
  """Compute binning statistics required for calibration measures.

  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n,nlabels), with probs for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.

  Returns:
    bz: Tensor, shape (2,num_bins), tf.int32, counts of incorrect (row 0) and
      correct (row 1) predictions in each of the `num_bins` probability bins.
    pmean_observed: Tensor, shape (num_bins,), tf.float32, the mean predictive
      probabilities in each probability bin.
  """
  if (logits is None) == (probabilities is None):
    raise ValueError(
        "_compute_calibration_bin_statistics expects exactly one of logits or "
        "probabilities.")
  if probabilities is None:
    if logits.get_shape().as_list()[-1] == 1:
      raise ValueError(
          "_compute_calibration_bin_statistics expects logits for binary"
          " classification of shape (n, 2) for nlabels=2 but got ",
          logits.get_shape())
    probabilities = tf.math.softmax(logits, axis=1)
  if (probabilities.get_shape().as_list()[-1] == 1 or
      len(probabilities.get_shape().as_list()) == 1):
    raise ValueError(
        "_compute_calibration_bin_statistics expects probabilities for binary"
        " classification of shape (n, 2) for nlabels=2 but got ",
        probabilities.get_shape())

  if labels_predicted is None:
    # If no labels are provided, we take the label with the maximum probability
    # decision.  This corresponds to the optimal expected minimum loss decision
    # under 0/1 loss.
    pred_y = tf.cast(tf.argmax(probabilities, axis=1), tf.int32)
  else:
    pred_y = labels_predicted

  correct = tf.cast(tf.equal(pred_y, labels_true), tf.int32)

  # Collect predicted probabilities of decisions
  prob_y = tf.compat.v1.batch_gather(probabilities,
                                     tf.expand_dims(pred_y, 1))  # p(pred_y | x)
  prob_y = tf.reshape(prob_y, (tf.size(prob_y),))

  # Compute b/z histogram statistics:
  # bz[0,bin] contains counts of incorrect predictions in the probability bin.
  # bz[1,bin] contains counts of correct predictions in the probability bin.
  bins = tf.histogram_fixed_width_bins(prob_y, [0.0, 1.0], nbins=num_bins)
  event_bin_counts = tf.math.bincount(
      correct*num_bins + bins,
      minlength=2*num_bins,
      maxlength=2*num_bins)
  event_bin_counts = tf.reshape(event_bin_counts, (2, num_bins))

  # Compute mean predicted probability value in each of the `num_bins` bins
  pmean_observed = tf.math.unsorted_segment_sum(prob_y, bins, num_bins)
  tiny = np.finfo(np.float32).tiny
  pmean_observed = pmean_observed / (
      tf.cast(tf.reduce_sum(event_bin_counts, axis=0), tf.float32) + tiny)

  return event_bin_counts, pmean_observed


def expected_calibration_error(num_bins, logits=None, probabilities=None,
                               labels_true=None,
                               labels_predicted=None):
  """Compute the Expected Calibration Error (ECE).

  This method implements equation (3) in [1].  In this equation the probability
  of the decided label being correct is used to estimate the calibration
  property of the predictor.

  Note: a trade-off exist between using a small number of `num_bins` and the
  estimation reliability of the ECE.  In particular, this method may produce
  unreliable ECE estimates in case there are few samples available in some bins.
  As an alternative to this method, consider also using
  `bayesian_expected_calibration_error`.

  #### References
  [1]: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
       On Calibration of Modern Neural Networks.
       Proceedings of the 34th International Conference on Machine Learning
       (ICML 2017).
       arXiv:1706.04599
       https://arxiv.org/pdf/1706.04599.pdf

  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n,nlabels), with probs for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.

  Returns:
    ece: Tensor, scalar, tf.float32.
  """
  if (logits is None) == (probabilities is None):
    raise ValueError(
        "expected_calibration_error expects exactly one of logits or "
        "probabilities.")
  if probabilities is None:
    if logits.get_shape().as_list()[-1] == 1:
      raise ValueError(
          "expected_calibration_error expects logits for binary"
          " classification of shape (n, 2) for nlabels=2 but got ",
          logits.get_shape())
    probabilities = tf.math.softmax(logits, axis=1)
  if (probabilities.get_shape().as_list()[-1] == 1 or
      len(probabilities.get_shape().as_list()) == 1):
    raise ValueError(
        "expected_calibration_error expects probabilities for binary"
        " classification of shape (n, 2) for nlabels=2 but got ",
        probabilities.get_shape())
  # Compute empirical counts over the events defined by the sets
  # {incorrect,correct}x{0,1,..,num_bins-1}, as well as the empirical averages
  # of predicted probabilities in each probability bin.
  event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
      num_bins, probabilities=probabilities,
      labels_true=labels_true,
      labels_predicted=labels_predicted)

  # Compute the marginal probability of observing a particular probability bin.
  event_bin_counts = tf.cast(event_bin_counts, tf.float32)
  bin_n = tf.reduce_sum(event_bin_counts, axis=0)
  pbins = bin_n / tf.reduce_sum(bin_n)  # Compute the marginal bin probability.

  # Compute the marginal probability of making a correct decision given an
  # observed probability bin.
  tiny = np.finfo(np.float32).tiny
  pcorrect = event_bin_counts[1, :] / (bin_n + tiny)

  # Compute the ECE statistic as defined in reference [1].
  ece = tf.reduce_sum(pbins * tf.abs(pcorrect - pmean_observed))

  return ece


def bayesian_expected_calibration_error(num_bins, logits=None,
                                        probabilities=None,
                                        labels_true=None,
                                        labels_predicted=None,
                                        num_ece_samples=500):
  """Sample from the posterior of the Expected Calibration Error (ECE).

  The Bayesian ECE is defined via a posterior over ECEs given the observed data.
  With a large number of observations it will closely match the ordinary ECE but
  with few observations it will provide uncertainty estimates about the ECE.

  This method produces iid samples from the posterior over ECE values and for
  practical use you can summarize these samples, for example by computing the
  mean or quantiles.  For example, the following code will compute a 10%-90%
  most probable region as well as the median ECE estimate.

  ```
  ece_samples = bayesian_expected_calibration_error(10, logits=logits,
                                                    labels=labels)
  tfp.stats.percentile(ece_samples, [10.0, 50.0, 90.0])
  ```

  Args:
    num_bins: int, number of probability bins, e.g. 10.
    logits: Tensor, (n,nlabels), with logits for n instances and nlabels.
    probabilities: Tensor, (n,nlabels), with probs for n instances and nlabels.
    labels_true: Tensor, (n,), with tf.int32 or tf.int64 elements containing
      ground truth class labels in the range [0,nlabels].
    labels_predicted: Tensor, (n,), with tf.int32 or tf.int64 elements
      containing decisions of the predictive system.  If `None`, we will use
      the argmax decision using the `logits`.
    num_ece_samples: int, number of posterior samples of the ECE to create.

  Returns:
    ece_samples: Tensor, (ece_samples,), tf.float32 elements, ECE samples.
  """
  if (logits is None) == (probabilities is None):
    raise ValueError(
        "bayesian_expected_calibration_error expects exactly one of logits or "
        "probabilities.")
  if probabilities is None:
    if logits.get_shape().as_list()[-1] == 1:
      raise ValueError(
          "bayesian_expected_calibration_error expects logits for binary"
          " classification of shape (n, 2) for nlabels=2 but got ",
          logits.get_shape())
    probabilities = tf.math.softmax(logits, axis=1)
  if (probabilities.get_shape().as_list()[-1] == 1 or
      len(probabilities.get_shape().as_list()) == 1):
    raise ValueError(
        "bayesian_expected_calibration_error expects probabilities for binary"
        " classification of shape (n, 2) for nlabels=2 but got ",
        probabilities.get_shape())

  event_bin_counts, pmean_observed = _compute_calibration_bin_statistics(
      num_bins, probabilities=probabilities,
      labels_true=labels_true, labels_predicted=labels_predicted)
  event_bin_counts = tf.cast(event_bin_counts, tf.float32)

  # Compute posterior over probability value in each bin
  bin_n = tf.reduce_sum(event_bin_counts, axis=0)
  post_ploc, post_pscale = _prob_posterior(num_bins, bin_n, pmean_observed)
  bin_centers = _bin_centers(num_bins)
  half_bin = 0.5*(1.0/num_bins)
  post_pmean = tfd.TruncatedNormal(
      loc=post_ploc, scale=post_pscale,
      low=bin_centers-half_bin, high=bin_centers+half_bin)

  # Compute the Dirichlet posterior over b/z probabilities
  prior_alpha = 1.0/num_bins  # Perk's Dirichlet prior
  post_alpha = tf.reshape(event_bin_counts + prior_alpha, (2*num_bins,))
  posterior_event = tfd.Dirichlet(post_alpha)

  # Sample ECEs from the analytic and posteriors, which factorize
  ece_samples = tf.stack(
      [_compute_ece(tf.reshape(posterior_event.sample(), (2, num_bins)),
                    post_pmean.sample()) for _ in range(num_ece_samples)])

  return ece_samples

