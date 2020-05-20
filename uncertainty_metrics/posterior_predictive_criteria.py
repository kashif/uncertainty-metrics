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

"""Information criteria evaluated on posterior predictive distributions.

The posterior predictive distribution of a model is the average of model
predictions weighted by the parameter posterior.  We implement criteria which
are generally applicable, including to singular models such as neural networks,
and which can be reliably estimated from Monte Carlo approximations to the
posterior distribution.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow.compat.v1 as tf  # tf


__all__ = [
    "negative_waic",
    "importance_sampling_cross_validation",
]


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

