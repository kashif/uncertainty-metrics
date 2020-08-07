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

"""Tests for AUC."""

import tensorflow as tf
import uncertainty_metrics as um


class AucTest(tf.test.TestCase):

  def test_AUC(self):
    m = um.AUC(num_thresholds=3)
    _ = m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
    # threshold values are [0 - 1e-7, 0.5, 1 + 1e-7]
    # tp = [2, 1, 0], fp = [2, 0, 0], fn = [0, 1, 2], tn = [0, 2, 2]
    # recall = [1, 0.5, 0], fp_rate = [1, 0, 0]
    # auc = ((((1+0.5)/2)*(1-0))+ (((0.5+0)/2)*(0-0))) = 0.75
    self.assertAlmostEqual(m.result().numpy(), 0.75)

if __name__ == '__main__':
  tf.test.main()
