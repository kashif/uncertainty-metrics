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

"""Information criteria.

The posterior predictive distribution of a model is the average of model
predictions weighted by the parameter posterior. We implement information
criteria for general predictive distributions and which can be reliably
estimated from Monte Carlo approximations to the posterior.
"""

import math
import tensorflow.compat.v1 as tf


def model_uncertainty(logits):
  """Mutual information between the categorical label and the model parameters.

  A way to evaluate uncertainty in ensemble models is to measure its spread or
  `disagreement`. One way is to measure the  mutual information between the
  categorical label and the parameters of the categorical output. This assesses
  uncertainty in predictions due to `model uncertainty`. Model
  uncertainty can be expressed as the difference of the total uncertainty and
  the expected data uncertainty:
  `Model uncertainty = Total uncertainty - Expected data uncertainty`, where

  * `Total uncertainty`: Entropy of expected predictive distribution.
  * `Expected data uncertainty`: Expected entropy of individual predictive
    distribution.

  This formulation was given by [1, 2] and allows the decomposition of total
  uncertainty into model uncertainty and expected data uncertainty. The
  total uncertainty will be high whenever the model is uncertain. However, the
  model uncertainty, the difference between total and expected data
  uncertainty, will be non-zero iff the ensemble disagrees.

  ## References:
  [1] Depeweg, S., Hernandez-Lobato, J. M., Doshi-Velez, F, and Udluft, S.
      Decomposition of uncertainty for active learning and reliable
      reinforcement learning in stochastic systems.
      stat 1050, p.11, 2017.
  [2] Malinin, A., Mlodozeniec, B., and Gales, M.
      Ensemble Distribution Distillation.
      arXiv:1905.00076, 2019.

  Args:
    logits: Tensor, shape (N, k, nc). Logits for N instances, k ensembles and
      nc classes.

  Raises:
    TypeError: Raised if both logits and probabilities are not set or both are
      set.
    ValueError: Raised if logits or probabilities do not conform to expected
      shape.

  Returns:
    model_uncertainty: Tensor, shape (N,).
    total_uncertainty: Tensor, shape (N,).
    expected_data_uncertainty: Tensor, shape (N,).
  """

  if logits is None:
    raise TypeError(
        "model_uncertainty expected logits to be set.")
  if tf.rank(logits).numpy() != 3:
    raise ValueError(
        "model_uncertainty expected logits to be of shape (N, k, nc),"
        "instead got {}".format(logits.shape))

  # expected data uncertainty
  log_prob = tf.math.log_softmax(logits, -1)
  prob = tf.exp(log_prob)
  expected_data_uncertainty = tf.reduce_mean(
      tf.reduce_sum(- prob * log_prob, -1), -1)

  n_ens = tf.cast(log_prob.shape[1], tf.float32)
  log_expected_probabilities = tf.reduce_logsumexp(
      log_prob, 1) - tf.math.log(n_ens)
  expected_probabilities = tf.exp(log_expected_probabilities)
  total_uncertainty = tf.reduce_sum(
      - expected_probabilities * log_expected_probabilities, -1)

  model_uncertainty_ = total_uncertainty - expected_data_uncertainty

  return model_uncertainty_, total_uncertainty, expected_data_uncertainty


def negative_waic(logp, waic_type="waic1"):
  """Compute the negative Widely Applicable Information Criterion (WAIC).

  The negative WAIC estimates the holdout log-likelihood from just the training
  data and an approximation to the posterior predictive.

  WAIC is a criterion that is evaluated on the _training set_ using the
  posterior predictive distribution derived from the _same_ training set, see
  [(Watanabe, 2018)][1].
  Because the posterior predictive distribution is typically not available in
  closed form, this implementation uses a Monte Carlo approximate,
  theta_j ~ p(theta | D), where D is the training data.

  Note that WAIC evaluated on the true parameter posterior is an accurate
  estimate to O(B^{-2}), however, in this implementation we have two additional
  sources of error: 1. the finite sample approximation to the posterior
  predictive, and 2. approximation error in the posterior due to approximate
  inference.

  For the rationale of why one would want to use WAIC, see [2].

  ### References:
  [1]: Sumio Watanabe. Mathematical Theory of Bayesian Statistics.
    CRC Press. 2018
    https://www.crcpress.com/Mathematical-Theory-of-Bayesian-Statistics/Watanabe/p/book/9781482238068
  [2]: Sebastian Nowozin.  Do Bayesians overfit?
    http://www.nowozin.net/sebastian/blog/do-bayesians-overfit.html


  Args:
    logp: Tensor, shape (B,M,...), containing log p(y_i | x_i, theta_j)
      for i=1,..,B instances and j=1,...,M models.
    waic_type: 'waic1' or 'waic2'.  The WAIC1 criterion uses the variance of the
      log-probabilities, the WAIC2 criterion uses the difference between the
      Bayes posterior and Gibbs posterior.

  Returns:
    neg_waic: Tensor, (...), the negative WAIC.
    neg_waic_sem: Tensor, (...), the standard error of the mean of `neg_waic`.
  """
  logp_mean = tf.reduce_logsumexp(logp, 1) - math.log(int(logp.shape[1]))
  if waic_type == "waic1":
    _, logp_var = tf.nn.moments(logp, 1)
    neg_waic, neg_waic_var = tf.nn.moments(logp_mean - logp_var, 0)
  elif waic_type == "waic2":
    gibbs_logp = tf.reduce_mean(logp, 1)
    neg_waic, neg_waic_var = tf.nn.moments(2.0*gibbs_logp - logp_mean, 0)

  neg_waic_sem = tf.sqrt(neg_waic_var / float(int(logp.shape[1])))

  return neg_waic, neg_waic_sem


def importance_sampling_cross_validation(logp):
  """Compute the importance-sampling cross validation (ISCV) estimate.

  The ISCV estimates the holdout log-likelihood from just an approximation to
  the posterior predictive log-likelihoods on the training data.

  ### References:
  [1]: Alan E. Gelfand, Dipak K. Dey, Hong Chang.
    Model determination using predictive distributions with implementation via
    sampling-based methods.
    Technical report No. 462, Department of Statistics,
    Stanford university, 1992.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.860.3702&rep=rep1&type=pdf
  [2]: Aki Vehtari, Andrew Gelman, Jonah Gabry.
    Practical Bayesian model evaluation using leave-one-out cross-validation and
    WAIC.
    arXiv:1507.04544
    https://arxiv.org/pdf/1507.04544.pdf
  [3]: Sumio Watanabe. Mathematical Theory of Bayesian Statistics.
    CRC Press. 2018
    https://www.crcpress.com/Mathematical-Theory-of-Bayesian-Statistics/Watanabe/p/book/9781482238068


  Args:
    logp: Tensor, shape (B,M,...), containing log p(y_i | x_i, theta_j)
      for i=1,..,B instances and j=1,...,M models.

  Returns:
    iscv_logp: Tensor, (...), the ISCV estimate of the holdout log-likelihood.
    iscv_logp_sem: Tensor, (...), the standard error of th emean of `iscv_logp`.
  """
  iscv_logp, iscv_logp_var = tf.nn.moments(tf.reduce_logsumexp(-logp, 1), 0)
  m = int(logp.shape[1])
  iscv_logp -= math.log(m)
  iscv_logp = -iscv_logp
  iscv_logp_sem = tf.sqrt(iscv_logp_var / float(m))

  return iscv_logp, iscv_logp_sem

