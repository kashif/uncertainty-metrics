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
"""Tests for general calibration error.
"""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import uncertainty_metrics as um


class GeneralCalibrationErrorTest(parameterized.TestCase, absltest.TestCase):

  def test_consistency(self):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374],
                      [0.44766216, 0.47721294, 0.0751249],
                      [0.1862702, 0.15139402, 0.66233578],
                      [0.05753544, 0.8561222, 0.08634236],
                      [0.18697925, 0.29836466, 0.51465609]])
    labels = np.array([0, 1, 2, 1, 2])
    calibration_error = um.gce(
        labels, probs, num_bins=30, binning_scheme='even',
        class_conditional=False, max_prob=True, norm='l1')
    self.assertAlmostEqual(calibration_error, 0.412713502)

  def test_binary_1d(self):
    probs = np.array([.91, .32, .66, .67, .57, .98, .41, .19])
    labels = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    calibration_error = um.gce(
        labels, probs, num_bins=30, binning_scheme='even',
        class_conditional=False, max_prob=True, norm='l1')
    self.assertAlmostEqual(calibration_error, 0.18124999999999997)

  def test_binary_2d(self):
    probs = np.array(
        [.91, .32, .66, .67, .57, .98, .41, .19]).reshape(8, 1)
    labels = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    calibration_error = um.gce(
        labels, probs, num_bins=30, binning_scheme='even',
        class_conditional=False, max_prob=True, norm='l1')
    self.assertAlmostEqual(calibration_error, 0.18124999999999997)

  def test_correctness(self):
    num_bins = 10
    pred_probs = [
        [0.31, 0.32, 0.27],
        [0.37, 0.33, 0.30],
        [0.30, 0.31, 0.39],
        [0.61, 0.38, 0.01],
        [0.10, 0.65, 0.25],
        [0.91, 0.05, 0.04],
    ]
    # max_pred_probs: [0.32, 0.37, 0.39, 0.61, 0.65, 0.91]
    # pred_class: [1, 0, 2, 0, 1, 0]
    labels = [1., 0, 0., 1., 0., 0.]
    n = len(pred_probs)

    # Bins for the max predicted probabilities are (0, 0.1), [0.1, 0.2), ...,
    # [0.9, 1) and are numbered starting at zero.
    bin_counts = [0, 0, 0, 3, 0, 0, 2, 0, 0, 1]
    bin_correct_sums = [0, 0, 0, 2, 0, 0, 0, 0, 0, 1]
    bin_prob_sums = [0, 0, 0, 0.32 + 0.37 + 0.39, 0, 0, 0.61 + 0.65, 0, 0, 0.91]

    correct_ece = 0.
    bin_accs = [0.] * num_bins
    bin_confs = [0.] * num_bins
    for i in range(num_bins):
      if bin_counts[i] > 0:
        bin_accs[i] = bin_correct_sums[i] / bin_counts[i]
        bin_confs[i] = bin_prob_sums[i] / bin_counts[i]
        correct_ece += bin_counts[i] / n * abs(bin_accs[i] - bin_confs[i])

    self.assertAlmostEqual(correct_ece,
                           um.ece([int(i) for i in labels],
                                  np.array(pred_probs)))

  def generate_params():  # pylint: disable=no-method-argument
    # 'self' object cannot be passes to parameterized.
    names = ['binning_scheme', 'max_probs', 'class_conditional',
             'threshold', 'norm']
    parameters = [['even', 'adaptive'], [True, False], [True, False],
                  [0.0, 0.01], ['l1', 'l2']]
    list(itertools.product(*parameters))
    count = 0
    dict_list = []
    for params in itertools.product(*parameters):
      param_dict = {}
      for i, v in enumerate(params):
        param_dict[names[i]] = v
      count += 1
      dict_list.append(param_dict)
    return dict_list

  @parameterized.parameters(generate_params())
  def test_generatable_metrics(self, class_conditional, threshold, max_probs,
                               norm, binning_scheme):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374],
                      [0.44766216, 0.47721294, 0.0751249],
                      [0.1862702, 0.15139402, 0.66233578],
                      [0.05753544, 0.8561222, 0.08634236],
                      [0.18697925, 0.29836466, 0.51465609]])

    labels = np.array([0, 1, 2, 1, 2])
    calibration_error = um.general_calibration_error(
        labels, probs, binning_scheme=binning_scheme, max_prob=max_probs,
        class_conditional=class_conditional, threshold=threshold, norm=norm)
    self.assertGreaterEqual(calibration_error, 0)
    self.assertLessEqual(calibration_error, 1)

if __name__ == '__main__':
  absltest.main()
