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

"""Implements semiparametric_calibration_error."""

#  This file incorporates work covered by the following copyright and
#  permission notice:
#
#      Copyright (c) 2019, 2020, Steve Yadlowsky
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#   IN THE SOFTWARE.

import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy.stats
from sklearn.model_selection import StratifiedKFold


class SemiparametricCalibrationError(object):
  """Class implementing Semiparametric Calibration Error."""

  def __init__(self, folds=5, weight_trunc=0.05, weights='constant',
               bootstrap_size=500, orthogonal=False, normalize=False,
               smoothing='kernel', hyperparam_attempts=50,
               default_hyperparam_range=None, verbose=False):
    # Folds are used for cross validation of hyperparameter (smoothness)
    # choices as well as cross fitting of semiparametric nuisance params.
    self.folds = folds
    self.kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=708)
    self.weight_trunc = weight_trunc
    self.orthogonal = orthogonal
    self.bootstrap_size = bootstrap_size
    self.verbose = verbose
    self.smoothing = smoothing
    self.normalize = normalize
    self.hyperparam_attempts = hyperparam_attempts
    self.weights = weights

    if default_hyperparam_range is None:
      # Use reasonable default hyperparam ranges validated in simulation. These
      # scale adaptive estimates made once basic properties of the data are
      # known.
      if smoothing == 'kernel':
        default_hyperparam_range = (0, 0.1)
      if smoothing == 'spline':
        default_hyperparam_range = (0.5, 1.1)

    self.default_hyperparam_range = default_hyperparam_range

  def _relative_weights(self, probs):
    return 1.0 / np.maximum(np.minimum(probs**2, (1 - probs)**2),
                            self.weight_trunc)

  def _max_power_weights(self, probs):
    return 1.0 / np.maximum(probs * (1 - probs), self.weight_trunc)

  def _constant_weights(self, probs):
    return np.ones(probs.shape[0])

  def weight_function(self, probs):
    if self.weights is None or self.weights == 'constant':
      return self._constant_weights(probs)
    elif self.weights == 'relative':
      return self._relative_weights(probs)
    elif self.weights == 'chi':
      return self._max_power_weights(probs)
    else:
      raise Exception('Weight function "{}" unknown'.format(self.weights))

  def _weighted_mean(self, samples, weights):
    if self.normalize:
      return np.mean(samples * weights) / np.mean(weights)
    else:
      return np.mean(samples * weights)

  def _weighted_se(self, samples, weights):
    n = samples.shape[0]
    boot_samps = np.zeros(self.bootstrap_size)
    for b in range(self.bootstrap_size):
      boot_idx = np.random.choice(n, n)
      boot_samps[b] = self._weighted_mean(samples[boot_idx], weights[boot_idx])
    return np.std(boot_samps)

  def _calculate_calibration_error(self, probs, labels, accs):
    """Compute L2 calibration error using semiparametric method."""
    # Given data and nuisance parameter estimates, compute parameter
    # estimate---the L2 calibration error---and it's standard error.
    stat = (probs - labels) * (probs - accs)
    if_ = (probs - accs) * (probs + accs - 2 * labels)

    if self.orthogonal:
      est = self._weighted_mean(if_, self.weight_function(probs))
    else:
      est = self._weighted_mean(stat, self.weight_function(probs))
    se = self._weighted_se(if_, self.weight_function(probs))

    return est, se

  def rms_calibration_error(self,
                            probs,
                            labels,
                            hyperparam_range=None):
    """Low bias estimate of L2 calibration error w/ smoothing instead of bins."""
    est, _ = self._calculate_calibration_error_crossfit(
        probs, labels, hyperparam_range=hyperparam_range)
    return est

  def rms_calibration_error_conf_int(self, probs, labels,
                                     hyperparam_range=None, alpha=0.05):
    """Confidence interval for L2 calibration error."""
    # Estimates L2 calibration error using a semi-parametric method that
    # reduces the bias of binning estimates of accuracy conditional on
    # confidence, and provides statistically valid confidence intervals for the
    # calibration error.
    est, se = self._calculate_calibration_error_crossfit(
        probs, labels, hyperparam_range=hyperparam_range)
    z_alpha_div_2 = -scipy.stats.norm.ppf(alpha / 2.0)
    return (np.sqrt(max(est - z_alpha_div_2 * se, 0)),
            np.sqrt(max(est, 0)),
            np.sqrt(max(est + z_alpha_div_2 * se, 0)))

  def _calculate_calibration_error_crossfit(self, probs, labels,
                                            hyperparam_range=None):
    """Compute calib error using optimal hyperparams for calibration function."""
    if hyperparam_range is None:
      if self.smoothing == 'spline':
        w = self.weight_function(probs)
        w /= np.mean(w)
        max_val = np.sum(w ** 2 * probs * (1-probs))
        scale_lower, scale_upper = self.default_hyperparam_range
        hyperparam_range = np.linspace(scale_lower * max_val,
                                       scale_upper * max_val,
                                       self.hyperparam_attempts)
      if self.smoothing == 'kernel':
        scale_lower, scale_upper = self.default_hyperparam_range
        hyperparam_range = (np.linspace(
            scale_lower * probs.shape[0] + 1, scale_upper * probs.shape[0],
            self.hyperparam_attempts) / (np.max(probs) - np.min(probs)))**2

    return self._calculate_calibration_error(
        probs, labels,
        self._calculate_opt_cross_fit_calibration_function(
            probs, labels, hyperparam_range))

  def _calculate_calibration_function(self, train_probs, train_labels,
                                      test_probs, sigma=1):
    """Compute smoothing estimate of calibration function."""
    # Calibration function is the expected accuracy conditional on the
    # confidence / probabilities outputted by the prediction model. This
    # fits a smoothing model for the accuracy on the train data and gets the
    # model predictions on the test_probs.
    weights = self.weight_function(train_probs)
    weights /= np.mean(weights)
    train_labels -= train_probs
    if self.smoothing == 'kernel':
      dists = np.abs(train_probs[np.newaxis, :] - test_probs[:, np.newaxis])
      kernel = np.exp(-sigma * (dists ** 2))
      preds = kernel.dot(train_labels) / kernel.sum(axis=1)
    elif self.smoothing == 'spline':
      order = np.argsort(train_probs)
      s = UnivariateSpline(train_probs[order], train_labels[order],
                           s=sigma, w=weights)
      preds = s(test_probs)
    else:
      raise Exception(
          'Smoothing type "{}" not implemented'.format(self.smoothing))
    preds += test_probs
    return preds

  def _calculate_cross_fit_calibration_function(self, probs, labels,
                                                hyperparams):
    """Helper function to estimate the calibration function w/ cross fitting."""
    accs = np.zeros(labels.shape)
    for train_index, test_index in self.kf.split(probs, labels):
      train_probs, test_probs = probs[train_index], probs[test_index]
      train_labels = labels[train_index]
      accs[test_index] = self._calculate_calibration_function(
          train_probs, train_labels, test_probs, hyperparams)
    return accs

  def _choose_opt_calibration_hyperparam(self, probs, labels, hyperparam_range):
    """Gets optimal prediction hyperparam from list of possibilies in hyperparam_range."""
    weights = self.weight_function(probs)
    weights /= np.mean(weights)

    best_error = np.float('inf')
    best_hyperparam = None
    for hyperparam in hyperparam_range:
      accs = self._calculate_cross_fit_calibration_function(
          probs, labels, hyperparam)
      error = np.mean(weights * (accs - labels) ** 2)
      if error < best_error:
        best_error = error
        best_hyperparam = hyperparam
    if self.verbose:
      print('Tried hyperparams: {}'.format(hyperparam_range))
      print('Best hyperparam: {}'.format(best_hyperparam))
    return best_hyperparam

  def _get_undersmoothed_hyperparam(self, probs, labels, hyperparam_range):
    """Adjust optimal hyperparam to work better for semiparametric estimation."""
    # The optimal hyperparams for prediction of accuracy with the calibration
    # function are generally more smooth than one wants when plugging them in to
    # a semiparametric estimator. This takes the optimal hyperparams for
    # prediction and fudges them a little bit to undersmooth by just the right
    # amount.
    n = labels.shape[0]
    opt_hyperparam = self._choose_opt_calibration_hyperparam(
        probs, labels, hyperparam_range)
    if self.smoothing == 'kernel':
      opt_hyperparam *= n ** 0.08
    else:
      # For now, use hacky adjustment, since it's not clear how to choose.
      # Should be slightly undersmoothed, but amount should depend on n.
      # In simulation, this works well for n between 1000 and 20000.
      opt_hyperparam *= 0.985
    return opt_hyperparam

  def _calculate_opt_cross_fit_calibration_function(
      self, probs, labels, hyperparam_range):
    """Get best, cross fit calibration function for semiparametric estimator."""
    hyperparam = self._get_undersmoothed_hyperparam(
        probs, labels, hyperparam_range)

    return self._calculate_cross_fit_calibration_function(
        probs, labels, hyperparam)


def semiparametric_calibration_error(
    probs, labels, hyperparam_range=None, **class_kwargs):
  """Estimate of L2 calibration error.

  Estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on
  confidence, and provides statistically valid confidence intervals for the
  calibration error.

  Args:
    probs: np.ndarray of shape [N, ] where N is the number of datapoints.
    labels: np.ndarray of shape [N, ] array of correct binary labels.
    hyperparam_range: np.ndarray of smoothing parameters to try when estimating
        the calibration function. If None is provided, will try to use
        reasonable defaults that worked well in simulation.
    **class_kwargs: dict to be provided to class construction with other
        hyperparameter options.

  Returns:
    Float, general calibration error.
  """
  ce = SemiparametricCalibrationError(**class_kwargs)
  return ce.rms_calibration_error(probs, labels, hyperparam_range)


def semiparametric_calibration_error_conf_int(
    probs, labels, hyperparam_range=None, **class_kwargs):
  """Confidence interval for L2 calibration error.

  Estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on
  confidence, and provides statistically valid confidence intervals for the
  calibration error.

  Args:
    probs: np.ndarray of shape [N, ] where N is the number of datapoints.
    labels: np.ndarray of shape [N, ] array of correct binary labels.
    hyperparam_range: np.ndarray of smoothing parameters to try when estimating
        the calibration function. If None is provided, will try to use
        reasonable defaults that worked well in simulation.
    **class_kwargs: dict to be provided to class construction with other
        hyperparameter options.

  Returns:
    Float, Lower CI on L2 calibration error.
    Float, Point estimate of L2 calibration error.
    Float, Upper CI on L2 calibration error.
  """
  ce = SemiparametricCalibrationError(**class_kwargs)
  return ce.rms_calibration_error_conf_int(probs, labels, hyperparam_range)


spce = semiparametric_calibration_error
spce_conf_int = semiparametric_calibration_error_conf_int
SPCE = SemiparametricCalibrationError
