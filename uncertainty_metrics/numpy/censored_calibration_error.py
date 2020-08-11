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

"""Implements censored_calibration_error."""

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
import scipy.stats
import sklearn.model_selection
import sklearn.utils


# pytype: disable=attribute-error
class StratifiedSurvivalKFold(sklearn.model_selection.StratifiedKFold):
  """Cross validation with survival outcomes."""
  # Idea thanks to
  # https://scottclowe.com/2016-03-19-stratified-regression-partitions/
  # Adapted for right censored survival outcomes by applying the above procedure
  # separately for censored and uncensored observations.

  def _iter_test_masks(self, times, events, groups=None):
    fold_ids = np.zeros(times.shape[0])

    event_times = times[events == 1]
    event_indices = np.where(events == 1)[0][np.argsort(event_times)]
    event_fold_ids = self._partition_sorted_data_frame(event_indices.shape[0],
                                                       self.n_splits)
    fold_ids[event_indices] = event_fold_ids

    if np.any(events == 0):
      censored_times = times[events == 0]
      censored_indices = np.where(events == 0)[0][np.argsort(censored_times)]
      censored_fold_ids = self._partition_sorted_data_frame(
          censored_indices.shape[0], self.n_splits)

      fold_ids[censored_indices] = censored_fold_ids

    for fold in range(self.n_splits):
      yield fold_ids == fold

  def _partition_sorted_data_frame(self, n, k):
    left_overs = n % k

    if self.shuffle:
      rng = sklearn.utils.check_random_state(self.random_state)
      fold_order = [rng.choice(k, k, replace=False) for _ in range(n // k)]
      fold_order.append(rng.choice(k, left_overs, replace=False))
    else:
      fold_order = [np.arange(k) for _ in range(n // k)]
      fold_order.append(np.arange(left_overs))

    return np.concatenate(fold_order)

  def split(self, times, events, groups=None):
    events = sklearn.utils.validation.check_array(
        events, ensure_2d=False, dtype=None)
    return super().split(times, events, groups)


class CensoredCalibrationError(object):
  """Semiparametric Calibration Error with Right-Censoring."""

  def __init__(self,
               end_time,
               folds=5,
               weight_trunc=0.05,
               weights='constant',
               survival_aggregation='Kaplan-Meier',
               kernel_fn='rbf',
               bootstrap_size=200,
               orthogonal=False,
               num_repeats=4,
               normalize=False,
               hyperparam_attempts=20,
               default_hyperparam_range=None,
               verbose=False):
    # Folds are used for cross validation of hyperparameter (smoothness)
    # choices as well as cross fitting of semiparametric nuisance params.
    self.folds = folds
    self.kf = StratifiedSurvivalKFold(
        n_splits=folds, shuffle=True, random_state=708)
    self.weight_trunc = weight_trunc
    self.orthogonal = orthogonal
    self.bootstrap_size = bootstrap_size
    self.verbose = verbose
    self.normalize = normalize
    self.hyperparam_attempts = hyperparam_attempts
    self.weights = weights
    self.end_time = end_time
    self.survival_aggregation = survival_aggregation
    self.kernel_fn = kernel_fn
    self.num_repeats = num_repeats

    if default_hyperparam_range is None:
      # Use reasonable default hyperparam ranges validated in simulation. These
      # scale adaptive estimates made once basic properties of the data are
      # known.
      default_hyperparam_range = (0, 0.1)

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

  def _get_times(self, observed_times):
    """Gets all unique times before self.end_time in sorted order."""
    end_time = self.end_time
    epsilon = 0
    if np.any(observed_times > end_time):
      epsilon = np.min(observed_times[observed_times > end_time]) - end_time
    # np.unique returns things in sorted order already so no need to sort.
    times = np.unique(observed_times)
    times = times[times <= end_time]
    if times[0] != 0:
      times = np.insert(times, 0, 0)
    if times[-1] != end_time:
      times = np.append(times, end_time)

    times = np.append(times, end_time + epsilon / 2)
    return times

  def _calculate_calibration_error(self, probs, study_times, events,
                                   local_outcome_process,
                                   local_censoring_process, local_at_risk):
    """Compute L2 calibration error using semiparametric method."""
    # Given data and nuisance parameter estimates, compute parameter
    # estimate---the L2 calibration error---and it's standard error.
    n = probs.shape[0]
    times = self._get_times(study_times)

    censoring_cum_hazard_increments, survival_curve_censoring, cond_mean,\
        compensator = self._get_calibration_function(local_outcome_process,
                                                     local_censoring_process,
                                                     local_at_risk)
    accs = cond_mean[:, 0]

    # Influence function of IPCW
    at_risk = np.array([y > times for y in study_times])
    at_risk_inclusive = np.array([y >= times for y in study_times])
    censoring_process = at_risk * (1 - events)[:, np.newaxis]
    censoring_incs = np.hstack(
        (np.zeros(n)[:, np.newaxis], -np.diff(censoring_process)))

    martingale_incs = censoring_incs - censoring_cum_hazard_increments * at_risk
    # prbl short for predictable
    prbl_survival_curve_censoring = np.hstack(
        (np.ones(n)[:, np.newaxis], survival_curve_censoring[:, 0:-1]))

    cond_mean[~at_risk] = 0
    compensator[~at_risk] = 0

    # TODO(yadlowsky): replace 0.001 with a set-able constant or adaptive value
    integrand = (probs[:, np.newaxis] -
                 cond_mean) * (probs - accs)[:, np.newaxis] / np.maximum(
                     prbl_survival_curve_censoring, 0.001)
    martingale_integral = (martingale_incs * integrand).sum(axis=1)

    # Calculate where events and censoring happens
    change_occ = study_times[:, np.newaxis] == times[np.newaxis, :]
    change_loc = np.where(change_occ)
    event_loc = np.where(
        np.logical_and((events == 1)[:, np.newaxis], change_occ))
    # This can be large (n_times x n_times) and we don't need it anymore.
    del change_occ

    # Influence function of recalibration function
    km_influence_denom = np.ones(n)
    km_influence_denom[change_loc[0]] = 1 / at_risk_inclusive[change_loc]
    s = np.sum(compensator[:, :-1] * at_risk[:, :-1], axis=1)
    compensator = (accs - probs) * (1 - accs) * (
        events * (study_times <= self.end_time) * km_influence_denom - s)

    # IPCW
    ipw_weights = np.zeros(n)
    ipw_weights[study_times > self.end_time] = 1 / survival_curve_censoring[
        study_times > self.end_time, -2]
    ipw_weights[event_loc[0]] = 1 / survival_curve_censoring[event_loc]
    if self.verbose:
      print(ipw_weights.max())
    ipw_weights = np.minimum(ipw_weights, 20)

    stat = ipw_weights * (probs - (study_times <= self.end_time)) * (
        probs - accs)

    if_ = ipw_weights * ((accs - probs) *
                         ((study_times <= self.end_time) -
                          probs)) + compensator - martingale_integral

    if self.orthogonal:
      est = self._weighted_mean(if_, self.weight_function(probs))
    else:
      est = self._weighted_mean(stat, self.weight_function(probs))
    se = self._weighted_se(if_, self.weight_function(probs))

    return est, se

  def rms_calibration_error(self,
                            probs,
                            times,
                            event,
                            hyperparam_range=None):
    """Low bias estimate of L2 calibration error w/ smoothing, not bins."""
    est, _ = self._calculate_calibration_error_crossfit(
        probs, times, event, hyperparam_range=hyperparam_range)
    return est

  def rms_calibration_error_conf_int(self, probs, times, event,
                                     hyperparam_range=None, alpha=0.05):
    """Confidence interval for L2 calibration error."""
    # Estimates L2 calibration error using a semi-parametric method that
    # reduces the bias of binning estimates of accuracy conditional on
    # confidence, and provides statistically valid confidence intervals for the
    # calibration error.
    est, se = self._calculate_calibration_error_crossfit(
        probs, times, event, hyperparam_range=hyperparam_range)
    z_alpha_div_2 = -scipy.stats.norm.ppf(alpha / 2.0)
    return (np.sqrt(max(est - z_alpha_div_2 * se, 0)),
            np.sqrt(max(est, 0)),
            np.sqrt(max(est + z_alpha_div_2 * se, 0)))

  def _calculate_calibration_error_crossfit(self, probs, study_times, events,
                                            hyperparam_range=None):
    """Compute calib error w/ optimal hyperparams for calibration function."""

    if hyperparam_range is None:
      scale_lower, scale_upper = self.default_hyperparam_range
      hyperparam_range = 1.0 / (np.linspace(
          scale_lower * probs.shape[0] + 1, scale_upper * probs.shape[0],
          self.hyperparam_attempts) / (np.max(probs) - np.min(probs)))**2

    # Because of the IPCW weights, it's important to run cross-fitting
    # multiple times and take the median.
    def _fitter(random_state_adjustment):
      random_state = self.kf.random_state
      self.kf.random_state = random_state + random_state_adjustment
      local_outcome_process, local_censoring_process, local_at_risk = \
          self._calculate_opt_cross_fit_processes(
              probs, study_times, events, hyperparam_range)
      self.kf.random_state = random_state
      return self._calculate_calibration_error(probs, study_times, events,
                                               local_outcome_process,
                                               local_censoring_process,
                                               local_at_risk)

    many_runs = np.array([_fitter(i) for i in range(self.num_repeats)])
    return tuple(np.nanmedian(many_runs, axis=0))

  def _kernel_smooth_over_time_axis(self, kernel, outcome, weights):
    """Computes kernel smoothed and weighted estimate of time series outcome."""
    return kernel.dot(weights[:, np.newaxis] * outcome) / kernel.dot(
        weights)[:, np.newaxis]

  def _nelson_aalen_aggregate(self, cumulative_hazard_increments):
    return np.exp(np.cumsum(-cumulative_hazard_increments, axis=1))

  def _kaplan_meier_aggregate(self, cumulative_hazard_increments):
    return np.cumprod(1 - cumulative_hazard_increments, axis=1)

  def _aggregate_survival_function(self, cumulative_hazard_increments):
    if self.survival_aggregation == 'Nelson-Aalen':
      return self._nelson_aalen_aggregate(cumulative_hazard_increments)
    elif self.survival_aggregation == 'Kaplan-Meier':
      return self._kaplan_meier_aggregate(cumulative_hazard_increments)
    else:
      raise 'Unsupported survival function aggregation "{}"'.format(
          self.survival_aggregation)

  def _calculate_local_processes(
      self, train_probs, train_time, train_event, times, test_probs,
      bandwidth=1):
    """Compute smoothing estimate of observed time series processes."""
    # Method here comes from the paper: Dabrowska, 1987, "Non-Parametric
    # Regression with Censored Survival Time Data". Scandinavian Journal of
    # Statistics.
    weights = self.weight_function(train_probs)
    weights /= np.mean(weights)
    at_risk = np.array([time > times for time in train_time])
    outcome_process = at_risk * train_event[:, np.newaxis]
    censoring_process = at_risk * (1-train_event)[:, np.newaxis]

    dists = np.abs(
        train_probs[np.newaxis, :] - test_probs[:, np.newaxis]) / bandwidth
    if self.kernel_fn == 'cubic':
      kernel = 15 * (1 - dists ** 2) ** 3 / 16
      kernel[dists > 1] = 0
    elif self.kernel_fn == 'rbf':
      kernel = np.exp(-dists ** 2)
    else:
      raise 'Kernel {} not supported'.format(self.kernel_fn)

    local_outcome_process = self._kernel_smooth_over_time_axis(
        kernel, outcome_process, weights)
    local_censoring_process = self._kernel_smooth_over_time_axis(
        kernel, censoring_process, weights)
    local_at_risk = self._kernel_smooth_over_time_axis(kernel, at_risk, weights)

    return local_outcome_process, local_censoring_process, local_at_risk

  def _get_calibration_function(self, local_outcome_process,
                                local_censoring_process, local_at_risk):
    """Compute smoothing estimate of calibration function."""
    # Calibration function is the expected accuracy conditional on the
    # confidence / probabilities outputted by the prediction model. This
    # converts smoothed estimates of the counting and at risk processes to the
    # calibration function and needed nuisance parameters for censoring adjusted
    # estimates of it.

    local_outcome_incs = -np.insert(np.diff(local_outcome_process, axis=1),
                                    0, 0, axis=1)
    local_censoring_incs = -np.insert(np.diff(local_censoring_process, axis=1),
                                      0, 0, axis=1)

    local_at_risk_minus = np.insert(local_at_risk, 0, 1, axis=1)[:, :-1]

    outcome_cum_hazard_increments = local_outcome_incs / local_at_risk_minus
    outcome_cum_hazard_increments[local_outcome_incs == 0] = 0
    censoring_cum_hazard_increments = local_censoring_incs / local_at_risk_minus
    censoring_cum_hazard_increments[local_censoring_incs == 0] = 0

    survival_curve_outcome = self._aggregate_survival_function(
        outcome_cum_hazard_increments)
    survival_curve_censoring = self._aggregate_survival_function(
        censoring_cum_hazard_increments)

    compensator = local_outcome_incs / (
        local_at_risk_minus * (local_at_risk_minus + local_outcome_incs))

    # Assumes last time step is self.end_time + \epsilon
    cond_mean = 1 - \
        survival_curve_outcome[:, -2][:, np.newaxis] / survival_curve_outcome

    return (censoring_cum_hazard_increments, survival_curve_censoring,
            cond_mean, compensator)

  def _calculate_cross_fit_processes(self, probs, study_times, events,
                                     bandwidth):
    """Helper function to estimate the calibration function w/ cross fitting."""
    unique_times = self._get_times(study_times)
    n_times = unique_times.shape[0]
    n = study_times.shape[0]

    local_outcome_process = np.zeros((n, n_times))
    local_censoring_process = np.ones((n, n_times))
    local_at_risk = np.zeros((n, n_times))

    for train_index, test_index in self.kf.split(study_times, events):
      print(test_index)
      train_probs, test_probs = probs[train_index], probs[test_index]
      train_study_times = study_times[train_index]
      train_events = events[train_index]

      pred_outcome_process, pred_censoring_process, pred_at_risk = \
          self._calculate_local_processes(train_probs, train_study_times,
                                          train_events, unique_times,
                                          test_probs, bandwidth)

      local_outcome_process[test_index, :] = pred_outcome_process
      local_censoring_process[test_index, :] = pred_censoring_process
      local_at_risk[test_index, :] = pred_at_risk

    return (local_outcome_process, local_censoring_process, local_at_risk)

  def _choose_opt_calibration_hyperparam(self, probs, time, event,
                                         hyperparam_range):
    """Gets optimal prediction hyperparam from list of hyperparam_range."""
    weights = self.weight_function(probs)
    weights /= np.mean(weights)

    unique_times = self._get_times(time)

    best_error = np.float('inf')
    best_hyperparam = None
    for hyperparam in hyperparam_range:
      local_outcome_process, local_censoring_process, _ = \
          self._calculate_cross_fit_processes(probs, time, event, hyperparam)

      observed_at_risk = np.array([y > unique_times for y in time])
      observed_outcome_process = observed_at_risk * event[:, np.newaxis]
      observed_censoring_process = observed_at_risk * (1 - event)[:, np.newaxis]

      # TODO(yadlowsky): consider re-weighting error by
      # 1 / local_censoring_process or whatever the right weighting is.
      error = np.mean(
          weights[:, np.newaxis] *
          ((observed_outcome_process - local_outcome_process)**2 +
           (observed_censoring_process - local_censoring_process)**2))
      if error < best_error:
        best_error = error
        best_hyperparam = hyperparam

    if self.verbose:
      print('Tried hyperparams: {}'.format(hyperparam_range))
      print('Best hyperparam: {}'.format(best_hyperparam))
    return best_hyperparam

  def _get_undersmoothed_hyperparam(self, probs, times, event,
                                    hyperparam_range):
    """Adjust optimal hyperparam to work for semiparametric estimation."""
    # The optimal hyperparams for prediction of accuracy with the calibration
    # function are generally more smooth than one wants when plugging them in to
    # a semiparametric estimator. This takes the optimal hyperparams for
    # prediction and fudges them a little bit to undersmooth by just the right
    # amount.
    n = times.shape[0]
    opt_hyperparam = self._choose_opt_calibration_hyperparam(
        probs, times, event, hyperparam_range)
    opt_hyperparam *= n**0.08
    return opt_hyperparam

  def _calculate_opt_cross_fit_processes(self, probs, study_times,
                                         event, hyperparam_range):
    """Get best, cross fit calibration function for semiparametric estimator."""
    hyperparam = self._get_undersmoothed_hyperparam(probs, study_times, event,
                                                    hyperparam_range)

    return self._calculate_cross_fit_processes(probs, study_times, event,
                                               hyperparam)
# pytype: enable=attribute-error


def censored_calibration_error(
    probs, times, event, end_time, hyperparam_range=None, **class_kwargs):
  """Estimate of L2 calibration error.

  Calibration error at level p is P(T <= end_time | probs = p) - p.
  Estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on
  confidence.

  Args:
    probs: np.ndarray of shape [N, ] where N is the number of datapoints.
    times: np.ndarray of shape [N, ] array of time that either the event or
        censoring occured.
    event: np.ndarray of shape [N, ] that indicates whether the event (1) or
        censoring (0) occured.
    end_time: float with the time at which the prediction should be calibrated.
    hyperparam_range: np.ndarray of smoothing parameters to try when estimating
        the calibration function. If None is provided, will try to use
        reasonable defaults that worked well in simulation.
    **class_kwargs: dict to be provided to class construction with other
        hyperparameter options.

  Returns:
    Float, general calibration error.
  """
  ce = CensoredCalibrationError(end_time, **class_kwargs)
  return ce.rms_calibration_error(probs, times, event, hyperparam_range)


def censored_calibration_error_conf_int(
    probs, times, event, end_time, hyperparam_range=None, **class_kwargs):
  """Confidence interval for L2 calibration error.

  Calibration error at level p is P(T <= end_time | probs = p) - p.
  Estimates L2 calibration error using a semi-parametric method that
  reduces the bias of binning estimates of accuracy conditional on
  confidence, and provides statistically valid confidence intervals for the
  calibration error.

  Args:
    probs: np.ndarray of shape [N, ] where N is the number of datapoints.
    times: np.ndarray of shape [N, ] array of time that either the event or
        censoring occured.
    event: np.ndarray of shape [N, ] that indicates whether the event (1) or
        censoring (0) occured.
    end_time: float with the time at which the prediction should be calibrated.
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
  ce = CensoredCalibrationError(end_time, **class_kwargs)
  return ce.rms_calibration_error_conf_int(probs, times, event,
                                           hyperparam_range)


cce = censored_calibration_error
cce_conf_int = censored_calibration_error_conf_int
CCE = CensoredCalibrationError
