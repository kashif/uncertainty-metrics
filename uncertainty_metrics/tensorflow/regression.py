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

"""Scoring rules suitable for regression.

Proper scoring rules allow us to evaluate of train probabilistic models with
real-valued outputs.
"""

import math
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp


__all__ = [
    "crps_score",
    "crps_normal_score",
]


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

