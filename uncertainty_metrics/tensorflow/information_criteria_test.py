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

"""Tests for information criteria."""

from absl import logging
from absl.testing import parameterized

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import uncertainty_metrics as um
tfd = tfp.distributions

eps = 1e-6


class EnsembleTest(tf.test.TestCase):

  def testEnsembleCrossEntropy(self):
    """Checks that ensemble cross entropy lower-bounds Gibbs cross entropy."""
    # For multi-class classifications
    batch_size = 2
    num_classes = 3
    ensemble_size = 5
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    logits = tf.random.normal([ensemble_size, batch_size, num_classes])
    ensemble_error = um.ensemble_cross_entropy(labels, logits)
    gibbs_error = um.gibbs_cross_entropy(labels, logits)
    self.assertEqual(ensemble_error.shape, ())
    self.assertEqual(gibbs_error.shape, ())
    self.assertLessEqual(ensemble_error, gibbs_error)

    # For binary classifications
    num_classes = 1
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.float32)
    logits = tf.random.normal([ensemble_size, batch_size, num_classes])
    loss_logits = tf.squeeze(logits, axis=-1)
    ensemble_error = um.ensemble_cross_entropy(labels, loss_logits, binary=True)
    gibbs_error = um.gibbs_cross_entropy(labels, loss_logits, binary=True)
    self.assertEqual(ensemble_error.shape, ())
    self.assertEqual(gibbs_error.shape, ())
    self.assertLessEqual(ensemble_error, gibbs_error)


class MutualInformationTest(tf.test.TestCase):

  def setUp(self, seed=42):
    super(MutualInformationTest, self).setUp()
    tf.set_random_seed(seed)
    self.logits = tf.random_normal((128, 32, 10))
    prob = tf.nn.softmax(self.logits)

    prob = tf.clip_by_value(prob, eps, 1 - eps)
    expected_data_uncertainty = tf.reduce_mean(
        tf.reduce_sum(- prob * tf.math.log(prob), -1), -1)

    expected_prob = tf.reduce_mean(prob, 1)
    total_uncertainty = tf.reduce_sum(
        - expected_prob * tf.math.log(expected_prob), -1)

    model_uncertainty = total_uncertainty - expected_data_uncertainty

    self.model_uncertainty = model_uncertainty
    self.total_uncertainty = total_uncertainty
    self.expected_data_uncertainty = expected_data_uncertainty

  def assertTensorsAlmostEqual(self, x, y, msg=None):
    abs_delta = tf.abs(tf.cast(x - y, tf.float32))
    msg += "(max delta: {:.4E})".format(tf.reduce_max(abs_delta).numpy())
    self.assertTrue(
        tf.reduce_all(abs_delta < eps), msg=msg)

  def test_model_uncertainty_with_logits(self):
    output, _, _ = um.model_uncertainty(logits=self.logits)
    msg = "Model uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(self.model_uncertainty, output, msg=msg)

  def test_total_uncertainty_with_logits(self):
    _, output, _ = um.model_uncertainty(logits=self.logits)
    msg = "Total uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(self.total_uncertainty, output, msg=msg)

  def test_expected_data_uncertainty_with_logits(self):
    _, _, output = um.model_uncertainty(logits=self.logits)
    msg = "Data uncertainty not almost equal what is expected."
    self.assertTensorsAlmostEqual(
        self.expected_data_uncertainty, output, msg=msg)

  def test_wrong_logits_shape(self):
    logits_wrong = tf.reduce_mean(self.logits, -1)
    with self.assertRaises(ValueError):
      um.model_uncertainty(logits=logits_wrong)


class PredictiveCriteriaTest(parameterized.TestCase, tf.test.TestCase):

  def _estimate_criterion_normal(self, mu0, sigma0, sigma, nsamples, nmodels,
                                 nsamples_opt, reps):
    """Create replicated experiments using a hierarchical Normal model.

    One experiment is conducted as follows:
      1. We samples mu ~ Normal(mu0, sigma0^2)
      2. For i = 1,..,nsamples, we sample x_i ~ Normal(mu, sigma^2)
      3. For i = 1,..,nsamples_opt, we sample y_i ~ Normal(mu, sigma^2)
      4. We compute analytically the posterior p(mu | x_1,..,x_nsamples)
      5. For j = 1,..,nmodels, we sample mu_j ~ p(mu | x_1,..,x_nsamples)
      6. We compute the predictive log-likelihoods log p(x_i | mu_j)
      7. The predictive training log-prob is the average over j
      8. We compute the optimal predictive log-likelihoods,
         log p(y_i | x_1,..,x_n).  This is possible because the optimal Bayes
         posterior predictive has a closed-form in the above model.

    Args:
      mu0: float, the prior mean.
      sigma0: float, >0.0, the prior standard deviation.
      sigma: float, >0.0, the observation model standard deviation.
      nsamples: int, >0, the number of samples to draw for training.
      nmodels: int, >0, the number of parameter posterior samples to use.
      nsamples_opt: int, >0, the number of data samples to use to compute the
        predictive log-likelihood under the optimal Bayes posterior predictive.
      reps: int, >0, the number of experimental replicates.

    Returns:
      logp: Tensor, (nsamples,nmodels,reps), predictive log-likelihoods.
        At logp[i,j,r] we have log p(x_{r,i} | mu_{r,j}), where r is the
        replicate index.
      logp_optimal: float, a Monte Carlo estimate of the average posterior
        predictive log-likelihood over all data and replicates.
      logp_training: float, an estimate of the average posterior predictive
        log-likelihood evaluated on the training sample.
    """
    prior_mu = tfd.Normal(mu0, sigma0)
    mu = prior_mu.sample(reps)
    px = tfd.Normal(mu, sigma*tf.ones(mu.shape))
    data = px.sample(nsamples)

    # Compute the posterior over mu and the posterior predictive; this has an
    # analytic solution, see
    # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    xmean = tf.reduce_mean(data, axis=[0])
    sigma_n_squared = 1.0 / (nsamples/(sigma**2.0) + 1.0/(sigma0**2.0))
    mu_n = sigma_n_squared*(mu0/(sigma0**2.0) + nsamples*xmean/(sigma**2.0))
    post_mu = tfd.Normal(mu_n, tf.sqrt(sigma_n_squared))
    post_pred = tfd.Normal(mu_n, tf.sqrt(sigma_n_squared + sigma**2.0))

    logp = list()
    for _ in range(nmodels):
      mu_j = post_mu.sample(1)
      pred_mu_j = tfd.Normal(mu_j, sigma*tf.ones(mu_j.shape))
      logp.append(pred_mu_j.log_prob(data))  # log p(x_i|mu_j), mu_j ~ p(mu|D)

    logp = tf.stack(logp, axis=1)

    data_holdout = px.sample(nsamples_opt)

    logp_optimal = float(tf.reduce_mean(post_pred.log_prob(data_holdout)))
    logp_training = float(tf.reduce_mean(logp))

    return logp, logp_optimal, logp_training

  def _estimate_criterion(self, criterion_name, logp):
    if criterion_name == "iscv":
      logp_crit, _ = um.importance_sampling_cross_validation(logp)
    elif criterion_name == "waic1":
      logp_crit, _ = um.negative_waic(logp, waic_type="waic1")
    elif criterion_name == "waic2":
      logp_crit, _ = um.negative_waic(logp, waic_type="waic2")

    # logp_crit is (reps,) Tensor, but we are interested in average over
    # experiments to check the expected properties of the criterion.
    logp_crit = float(tf.reduce_mean(logp_crit))

    return logp_crit

  @parameterized.named_parameters(
      ("iscv_criterion", "iscv"),
      ("waic1_criterion", "waic1"),
      ("waic2_criterion", "waic2"))
  def test_criterion(self, criterion_name):
    mu0 = 0.0
    sigma0 = 1.0
    sigma = 0.5
    reps = 1000
    nsamples = 20
    nsamples_opt = 50000
    nmodels = 800

    logp, logp_optimal, logp_training = self._estimate_criterion_normal(
        mu0, sigma0, sigma, nsamples, nmodels, nsamples_opt, reps)
    logp_crit = self._estimate_criterion(criterion_name, logp)

    logging.info("%s predicts %.5f generalization loss, optimal is %.5f, "
                 "training loss is %.5f", criterion_name, logp_crit,
                 logp_optimal, logp_training)

    # Check that the training loss is smaller than the holdout Bayes loss.
    # This will stochastically always be the case.  Here error = -log_prob.
    self.assertLess(logp_optimal, logp_training, msg="Optimal predictive log "
                    "prob exceeds training log likelihood.")

    # Check that the information criteria is a good predictor of the holdout
    # Bayes loss by being closer to the Bayes loss than the training loss
    opt_crit_diff = abs(logp_optimal - logp_crit)
    opt_training_diff = abs(logp_optimal - logp_training)
    self.assertLess(opt_crit_diff, opt_training_diff,
                    msg="Training log prob is closer (%f) to optimal "
                    "predictive log prob than information criteria is close "
                    "to optimal predictive log prob (%f)" %
                    (opt_training_diff, opt_crit_diff))

    # Check that there is a gap between the training error and the Bayes error.
    self.assertNotAlmostEqual(logp_optimal, logp_training, places=2,
                              msg="Training (%f) and generalization (%f) "
                              "error too close." %
                              (logp_training, logp_optimal))


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
