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

"""Tests for visualization."""

from absl.testing import absltest
import numpy as np
import uncertainty_metrics.numpy as um


class VisualizationTest(absltest.TestCase):

  def test_reliability_diagram(self):
    probs = np.array([[0.42610548, 0.41748077, 0.15641374, 0],
                      [0.44766216, 0.47721294, 0.0751249, 0],
                      [0.1862702, 0.15139402, 0.66233578, 0],
                      [0.05753544, 0.8561222, 0.08634236, 0],
                      [0.18697925, 0.29836466, 0.51465609, 0]])

    labels = np.array([0, 1, 2, 1, 2])
    rd = um.reliability_diagram(labels, probs)
    self.assertGreater(rd.get_figwidth(), 1)

  def test_binary_1d(self):
    probs = np.array([np.random.rand() for _ in range(10000)])
    labels = np.array([np.random.binomial(1, p=p) for p in probs])
    rd = um.reliability_diagram(labels, probs)
    self.assertGreater(rd.get_figwidth(), 1)

  def test_binary_2d(self):
    probs = np.array([np.random.rand() for _ in range(10000)]).reshape(10000, 1)
    labels = np.array([np.random.binomial(1, p=p) for p in probs])
    rd = um.reliability_diagram(labels, probs)
    self.assertGreater(rd.get_figwidth(), 1)

  def test_confidence_vs_accuracy_diagram(self):
    in_probs = np.array([np.random.rand() for _ in range(10000)])
    in_labels = np.array([np.random.binomial(1, p=p) for p in in_probs])
    out_probs = np.array([np.random.rand() for _ in range(10000)])

    acd = um.plot_confidence_vs_accuracy_diagram(in_probs, in_labels, out_probs)
    self.assertGreater(acd.figure.get_figwidth(), 1)

  def test_rejection_classification_diagram(self):
    in_probs = np.array([np.random.rand() for _ in range(10000)])
    in_labels = np.array([np.random.binomial(1, p=p) for p in in_probs])
    out_probs = np.array([np.random.rand() for _ in range(10000)])

    acd = um.plot_rejection_classification_diagram(
        in_probs, in_labels, out_probs)
    self.assertGreater(acd.figure.get_figwidth(), 1)

if __name__ == '__main__':
  absltest.main()
