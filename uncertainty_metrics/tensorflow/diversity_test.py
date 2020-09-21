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

"""Tests for diversity metrics."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import uncertainty_metrics as um


class DiversityTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (um.cosine_distance,),
      (um.disagreement,),
      (um.logit_kl_divergence,),
      (um.kl_divergence,),
      (um.lp_distance,),
  )
  def testDiversityMetrics(self, metric_fn):
    # TODO(trandustin): Test shapes. Need to change API to make it consistent.
    batch_size = 2
    num_classes = 3
    logits_one = tf.random.normal([batch_size, num_classes])
    logits_two = tf.random.normal([batch_size, num_classes])
    _ = metric_fn(logits_one, logits_two)

  def testAveragePairwiseDiversity(self):
    num_models = 3
    batch_size = 2
    num_classes = 5
    logits = tf.random.normal([num_models, batch_size, num_classes])
    probs = tf.nn.softmax(logits)
    results = um.average_pairwise_diversity(probs, num_models=num_models)
    self.assertLen(results, 3)
    self.assertEqual(results['disagreement'].shape, [])
    self.assertEqual(results['average_kl'].shape, [])
    self.assertEqual(results['cosine_similarity'].shape, [])

  def testVarianceBound(self):
    probs = np.array([
        [
            [.5, .4, .1],  # Model 1
            [.1, .1, .8]
        ],
        [
            [.3, .3, .4],  # Model 2
            [.1, 0., .9]
        ],
        [
            [.8, .1, .1],  # Model 3
            [0., .6, .3]
        ],
        [
            [.5, .1, .1],  # Model 4
            [.1, .2, .7]
        ],
    ])
    num_models, batch_size, _ = probs.shape  # 4, 2, 3.
    labels = tf.convert_to_tensor([0, 2], dtype=tf.int32)

    max_probs = [.8, .9]
    avg_probs = [.52, .675]

    expected_upper_bound = 0
    for i in range(batch_size):
      for j in range(num_models):
        label = labels[i]
        prob = probs[j, i, label]
        avg_prob = avg_probs[i]
        max_prob = max_probs[i]
        expected_upper_bound += ((prob - avg_prob) / max_prob)**2

    expected_upper_bound /= (2 * num_models * batch_size)
    actual_upper_bound = um.variance_bound(probs, labels, num_models=4)
    self.assertAllClose(actual_upper_bound, expected_upper_bound, atol=1e-5)


if __name__ == '__main__':
  tf.test.main()
