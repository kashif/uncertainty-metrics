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

"""Wraps AUC-PR and AUC-ROC.

Find source at https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC.
"""

import tensorflow.compat.v2 as tf


class AUC(tf.keras.metrics.AUC):
  """Computes the approximate AUC (Area under the curve) via a Riemann sum.

  This metric creates four local variables, true_positives, true_negatives,
  false_positives and false_negatives that are used to compute the AUC. To
  discretize the AUC curve, a linearly spaced set of thresholds is used to
  compute pairs of recall and precision values. The area under the ROC-curve
  is therefore computed using the height of the recall values by the false
  positive rate, while the area under the PR-curve is the computed using the
  height of the precision values by the recall.

  This value is ultimately returned as auc, an idempotent operation that
  computes the area under a discretized curve of precision versus recall
  values (computed using the aforementioned variables). The num_thresholds
  variable controls the degree of discretization with larger numbers of
  thresholds more closely approximating the true AUC. The quality of the
  approximation may vary dramatically depending on num_thresholds. The
  thresholds parameter can be used to manually specify thresholds which split
  the predictions more evenly.

  For best results, predictions should be distributed approximately uniformly
  in the range [0, 1] and not peaked around 0 or 1. The quality of the AUC
  approximation may be poor if this is not the case. Setting summation_method
  to 'minoring' or 'majoring' can help quantify the error in the approximation
  by providing lower or upper bound estimate of the AUC.

  If sample_weight is None, weights default to 1. Use sample_weight of 0 to
  mask values.

  """
  pass
