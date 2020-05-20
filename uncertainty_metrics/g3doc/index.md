# Prediction Metrics

This project implements several metrics to measure the prediction
quality of probabilistic models that are implemented using TensorFlow.

There are two uses of predictive metrics:

1.  _Model evaluation_: assessing the quality of your probabilistic model on a
    validation or test set.
2.  _Model training_: you can use the loss functions defined in this project to
    train probabilistic classification and regression models.

To use it, import the desired submodule. For example, in order to use the
regression scores, use

```python
from uncertainty_metrics import regression
```

## Table of Contents

[TOC]

## Key Concepts

### Proper Scoring Rules {#proper-scoring-rules}

_Proper scoring rules_ are loss functions for probabilistic predictions.
Formally, a proper scoring rule is a function which assign a numerical score
$$S(P,y)$$ to a predicted distribution $$P$$ and a realized value $$y$$.
Assuming that data instances are generated according to an unknown distribution,
$$(x_i,y_i) \sim Q$$, we would evaluate $$S(P(y|x_i),y_i)$$.

Common examples of proper scoring rules include the log-likelihood, the Brier
score, and the continuous ranked probability score (CRPS).

__Reference__

* _Tilmann Gneiting_ and _Adrian E. Raftery_, "Strictly Proper Scoring Rules,
  Prediction, and Estimation", Journal of the American Statistical Association
  (JASA), 2007.
  [(PDF)](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)

### Calibration {#calibration}

_Calibration_ refers to a frequentist property of probabilistic predictions being correct _on average_.  Intuitively, when predicting a binary outcome using a model, if we group all predictions where the outcome is believed to be 80 percent probable, then within this group we should, on average, see this outcome become true 80 percent of the time.  A model with this property is said to be well-calibrated.

Formally, calibration can best be defined in terms of measuring the difference between a conditional predictive distribution and the true conditional distribution, where
the conditioning is done with respect to a set defined as a function of the prediction.
For $$(x,y) \sim Q$$, with $$x \in \mathcal{X}$$, we consider $$\gamma = f(x)$$ a function of the prediction.
For example, $$\gamma$$ could be the vector of logits of a classifier, or a function thereof.  (In practice it will often be a discretization of a continuous quantity.)

Then, the _true conditional distribution_ $$q(y|\gamma)$$ corresponds to the true distribution over $$y$$ over the subset
$$\mathcal{X}_{\gamma} := \{x \in \mathcal{X} \,|\, f(x) = \gamma\}$$, i.e.

$$q(y|\gamma) =
\frac{\mathbb{E}_{x \sim Q}[\mathbb{1}_{\{f(x)=\gamma\}} \, q(y|x)]}{Q(\mathcal{X}_{\gamma})}.$$

The _model predictive conditional distribution_ $$p(y|\gamma)$$ is likewise given as

$$p(y|\gamma) =
\frac{\mathbb{E}_{x \sim Q}[\mathbb{1}_{\{f(x)=\gamma\}} \, p(y|x)]}{Q(\mathcal{X}_{\gamma})}.$$

The _reliability_ of a probabilistic prediction system is now defined as an expected difference between the two quantities,

$$\textrm{Reliability} = \mathbb{E}_{(x,y) \sim Q}[
D(q(y|\gamma(x)) \,\|\, p(y|\gamma(x)))].$$

A reliability of zero means that the model is perfectly calibrated: the predictions are on average correct.
In practice the reliability needs to be estimated: none of the expectations are available analytically.  Most importantly, to estimate the true conditional distribution requires discretization of the set of predictions.

__Reference__

* _Jochen Brocker_, "Reliability, sufficiency, and the decomposition of proper scores",
  Quarterly Journal of the Royal Meteorological Society, 2009.
  [(PDF)](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.456)

### Mutual Information {#mutual-information}

Mutual information is a way to measure the spread or _disagreement_ of an
ensemble $$\{p(y|x; \theta^{(m)})\}^M_{m=1}$$. These metrics are used after or
during model training to estimate the predictive performance on the
hold-out/test set. We differ between two mutual information based metrics,
_model uncertainty_ and _knowledge uncertainty_. Model uncertainty estimates the
mutual information between the categorical label $$y$$ and the model parameters
$$\theta$$, whereas knowledge uncertainty is the mutual information between the
categorical label $$y$$ and parameters $$\pi$$ of the categorical distribution.
Both model and knowledge uncertainty can be expressed as the difference of the
total uncertainty and the expected data uncertainty, where total uncertainty is
the entropy of expected predictive distribution and expected data uncertainty is
the expected entropy of individual predictive distribution.

Formally, given an ensemble model $$\{p(y|x^*,\theta^{(m)})\}_{m=1}^M$$ trained
on a finite dataset $$D$$, model uncertainty for a test input $$x^*$$ is defined
as:

$$ MI[y, \theta|x^*, D] =
\mathcal{H}(\mathbb{E}_{p(\theta|D)}[p(y|x^*,\theta)]) -
\mathbb{E}_{p(\theta|D)}[\mathcal{H}(p(u|x^*, \theta))] $$

The total uncertainty will be high whenever the model is uncertain - both in
regions of severe class overlap and out-of-domain. However, model uncertainty,
the difference between total and expected data uncertainty, will be non-zero iff
the ensemble disagrees.

Knowledge uncertainty estimates the mutual information between between the
categorical output $$y$$ and the parameters $$\pi$$ of the categorical
distribution. Malinin et al. introduced this metric specifically for ensemble
distribution distillation. It behaves exactly the same way as model uncertainty,
but the spread is now explicitly due to distributional uncertainty, rather than
model uncertainty. Given $$p(\pi|x^*, \hat{\theta}) = \textrm{Dir}(\pi|\alpha)$$
which models a Dirichlet distribution over the categorical output distribution
$$\pi$$, knowledge uncertainty is defined as:

$$
MI[y, \pi|x^*, \hat{\theta}] =
\mathcal{H}(\mathbb{E}_{p(\pi| x^*, \hat{\theta})}[p(y|\pi)]) -
\mathbb{E}_{p(\pi|x^*,\hat{\theta})}[\mathcal{H}(p(y|\pi))]
$$

__Reference__

*   _Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, and
    Steffen Udluft_, "Decomposition of uncertainty for active learning and
    reliable reinforcement learning in stochastic systems", stat 1050, 2017.
    [(PDF)](https://proceedings.mlr.press/v80/depeweg18a/depeweg18a.pdf)

*   _Andrey Malinin, Bruno Mlodozeniec and Mark Gales_, "Ensemble Distribution
    Distillation.", arXiv:1905.00076, 2019.
    [(PDF)](https://arxiv.org/pdf/1905.00076.pdf)

### Information Criteria {#information-criteria}

_Information criteria_ are used after or during model training to estimate the predictive performance on future holdout data.  They can be useful for selecting among multiple possible models or to perform hyperparameter optimization.  There are also strong connections between [cross validation estimates](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) and some information criteria.

__Reference__

* _Sumio Watanabe_, "Mathematical Theory of Bayesian Statistics",
  CRC Press, 2018.

## API: Scoring Rules

### Brier score {#brier-score}

The Brier score for discrete labels is defined as follows: given a predicted
probability vector $$P = (p_1,p_2,\dots,p_L)$$ and given a realized value $$y
\in \{1,2,\dots,L\}$$ the brier score is

$$S_{\textrm{Brier}}(P,y) = -2 p_y + \sum_{i=1}^L p_i^2.$$

Here is an example of how to use the Brier score as loss function for a
classifier. Suppose you have a classifier implemented in Tensorflow and your
current training code looks like

```python
  per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=target_labels, logits=logits)
  loss = tf.reduce_mean(per_example_loss)
```

Then you can alternatively use the API-compatible [_Brier loss_](#brier-score)
as follows:

```python
  per_example_loss = predictive_metrics.calibration.brier_score(
    labels=target_labels, logits=logits)
  loss = tf.reduce_mean(per_example_loss)
```

The Brier score penalizes low-probability predictions which do occur less than
the cross-entropy loss.

### Continuous Ranked Probability Score (CRPS) {#crps}

The continuous ranked probability score (CRPS) has several equivalent
definitions.

_Definition 1_: CRPS measures the integrated squared difference between an
arbitrary cummulative distribution function $$F$$ and a
[Heaviside distribution function](https://en.wikipedia.org/wiki/Heaviside_step_function)
at the realization, $$y$$,

$$S_{\textrm{CRPS}}(F,y) = \int_{-\infty}^{\infty}
(F(z) - \mathbb{1}_{\{z \geq y\}})^2 \,\textrm{d}z.$$

_Definition 2_: CRPS measures the expected distance to the realization minus one
half the expected distance between samples,

$$S_{\textrm{CRPS}}(F,y) = \mathbb{E}_{z \sim F}[|z-y|]
- \frac{1}{2} \mathbb{E}_{z,z' \sim F}[|z-z'|].$$

CRPS has two desirable properties:

1. It generalizes the absolute error loss and recovers the absolute error if a predicted distribution $$F$$ is deterministic.
2. It is reported in the same units as the predicted quantity.

To compute CRPS we either need to make an assumption regarding the form of $$F$$
or need to approximate the expectations over $$F$$ using samples from the
predictive model. In the current code we implement one analytic solution to CRPS
for predictive models with univariate Normal predictive distributions, and one
generic form for univariate predictive regression models that uses a sample
approximation.

__Normal model__

For a regression model which predicts Normal distributions with mean `means` and
standard deviation `stddevs`, we can compute CRPS as follows,

```python
  squared_errors = tf.square(target_labels - pred_means)
  per_example_crps = predictive_metrics.regression.crps_normal_score(
    labels=target_labels,
    means=pred_mean,
    stddevs=pred_stddevs)
```

__Sample-based approximation__

As long as we can sample predictions from our model, we can construct a Tensor
`predictive_samples` of size `(ninstances, npredictive_samples)` and evaluate the
Monte Carlo CRPS against the true targets `target_labels` using the following
code,

```python
  per_example_crps = predictive_metrics.regression.crps_score(
    labels=target_labels,
    predictive_samples=predictive_samples)
```

## API: Calibration Metrics

### Brier decomposition {#brier-decomposition}

Here is an example of how to compute calibration metrics for a classifier.
Suppose you evaluate the accuracy of your classifier on the validation set,

```python
  logits = model(validation_data)
  class_prediction = tf.argmax(logits, 1)
  accuracy = tf.metrics.accuracy(validation_labels, class_prediction)
```

You can compute additional metrics using the so called
[Brier decomposition](#brier-decomposition) that quantify prediction
_uncertainty_, _resolution_, and _reliability_ by appending the following code,

```python
  uncert, resol, reliab = predictive_metrics.calibration.brier_decomposition(
    labels=validation_labels, logits=logits)
```

In particular, the reliability (`reliab` in the above line) is a measure of
calibration error, with a value between 0 and 2, where zero means perfect
calibration.

### Expected Calibration Error {#ece}

The expected calibration error (ECE) is a scalar summary statistic between zero
(perfect calibration) and one (fully miscalibrated).
It is widely reported in the literature as a standard summary statistic
for classification.

The ECE metric works by binning the probability of the _decision label_ into a
fixed number of `num_bins` bins.  Typically `num_bins` is chosen to be 5 or 10.
For example, a binning into 5 bins would yield a partition,

$$\{(-\infty,0.2], (0.2,0.4], (0.4,0.6], (0.6,0.8], (0.8,\infty)\},$$

and all counting and frequency estimation of probabilities is done using this
binning.
For $$M$$ being `num_bins`, and a total of $$n$$ samples, the ECE is defined as

$$\textrm{ECE} := \sum_{m=1}^{M}
\frac{|B_m|}{n} |\textrm{acc}(B_m) - \textrm{conf}(B_m)|,$$

where $$B_m$$ is the set of instances with predicted probability
$$\hat{p}_i(\hat{y}_i)$$ of the decision label $$\hat{y}_i$$ assigned to
bin $$m$$, and with the true label $$y_i$$, using

$$\textrm{acc}(B_m) :=
\frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}_{\{\hat{y}_i=y_i\}},$$

and

$$\textrm{conf}(B_m) :=
\frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i(\hat{y}_i).$$

To compute the ECE metric you can pass in the decision labels $$\hat{y}_i$$
using the `labels_predicted` keyword argument.
In case you do not use `labels_predicted`,
the argmax label will be automatically inferred from the `logits`.  (Therefore,
in the following code we could remove `labels_predicted`.)

```python
  # labels_true is a tf.int32 Tensor
  logits = model(validation_data)
  labels_predicted = tf.argmax(logits, axis=1)
  ece = predictive_metrics.calibration.expected_calibration_error(
      10, logits=logits, labels_true=labels_true,
      labels_predicted=labels_predicted)
```

Reference:

* _Chuan Guo_, _Geoff Pleiss_, _Yu Sun_, and _Kilian Q. Weinberger_,
  ["On Calibration of Modern Neural Networks"](https://arxiv.org/pdf/1706.04599.pdf),
  _Proceedings of the 34th International Conference on Machine Learning_
  (ICML 2017), [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)


### Bayesian Expected Calibration Error {#bayesian-ece}

The [ECE metric](#ece) is a scalar summary statistic of miscalibration
evaluated on a finite sample of validation data.  Resulting in a single scalar,
the sampling variation due to the limited amount of validation data is hidden,
and this can result in significant over- or under-estimation of the ECE as well
as wrongly concluding significant differences in ECE between multiple models.

To address these issues, a Bayesian estimator of the ECE can be used.  The
resulting estimate is not a single scalar summary but instead a probability
distribution over possible ECE values.

Using the probability distribution of ECE values we can make statistical
conclusions.
For example, we can assess whether the ECE really is significantly different
between two or more models by reasoning about the overlap of the
ECE distributions of each model, using for example a Wilcoxon signed rank test.

The Bayesian ECE can be [used like the normal ECE](#ece), as in the following
code:

```python
  # labels_true is a tf.int32 Tensor
  logits = model(validation_data)
  ece_samples = predictive_metrics.calibration.bayesian_expected_calibration_error(
      10, logits=logits, labels_true=labels_true)

  ece_quantiles = tensorflow_probability.stats.percentile(
    ece_samples, [10,50,90])
```

The above code also includes an example of using the samples to infer
10%/50%/90% quantiles of the distribution of possible ECE values.

[Further details regarding the Bayesian model used to estimate ECE are
available](bayesian-ece.md).

## API: Mutual Information

### Model Uncertainty

Model uncertainty measures the mutual information between the categorical output
$$y$$ and the model parameters $$\theta$$. It is the difference of the total
uncertainty and expected data uncertainty between $$y$$ and $$\theta$$. The
implementation uses the logits to calculate model uncertainty. It returns
knowledge uncertainty as well as total uncertainty and expected data
uncertainty. An example code is given below:

```python
  logits = model(validation_data)
  model_uncert, total_uncert, avg_data_uncert = mutual_information.model_uncertainty(
      logits)
```

## API: Information Criteria

We estimate information criteria using log-likelihood values on training samples.
In particular, for both the WAIC and the ISCV criteria we assume that we have an ensemble of models with equal weights, such that the average predictive distribution over the ensemble is a good approximation to the true Bayesian posterior predictive distribution.

For a set of $$n$$ training instances and $$m$$ ensemble models we will use a
Tensor of shape $$(n,m)$$, such that the `[i,j]` element contains
$$\log p(y_i | x_i, \theta_j)$$, where $$\theta_j$$ represents the `j`'th ensemble element.  For example, the $$\theta_j$$ values can be obtained by MCMC sampling or as samples from a variational posterior distribution over parameters.

If the $$\theta$$ samples closely approximate the true posterior over parameters, then the WAIC and ISCV information criteria have strong theoretical properties in that they consistently predict the generalization loss with low variance.

### Widely/Watanabe Applicable Information Criterion (WAIC)

The negative WAIC criterion estimates log-likelihood on future observables.
It is given as

$$\textrm{nWAIC}_1 :=
\frac{1}{n} \sum_{i=1}^n \left(
  \log \frac{1}{m} \sum_{j=1}^m p(y_i|x_i,\theta_j)
  - \hat{\mathbb{V}}_i
\right),$$

where $$\hat{\mathbb{V}}_i := \frac{1}{m-1} \sum_{j=1}^m \left(
\log p(y_i|x_i,\theta_j) - \frac{1}{m} \sum_{k=1}^m \log p(y_i|x_i,\theta_k)\right)^2$$.

An alternative form of the WAIC, called type-2 WAIC, is computed as

$$\textrm{nWAIC}_2 :=
\frac{1}{n} \sum_{i=1}^n \left(
  \frac{2}{m} \sum_{j=1}^m \log p(y_i|x_i,\theta_j)
  - \log \frac{1}{m} \sum_{j=1}^m p(y_i|x_i,\theta_j)
\right).$$

Both nWAIC criteria have comparable properties, but Watanabe recommends $$\textrm{nWAIC}_1$$.
To estimate the negative WAIC, we use the following code.

```python
  import predictive_metrics.posterior_predictive_criteria as ppc

  # logp has shape (n,m), n instances, m ensemble members
  neg_waic, neg_waic_sem = ppc.negative_waic(logp, waic_type="waic1")
```

You can select the type of nWAIC to estimate using `waic_type="waic1"` or
`waic_type="waic2"`.  The method returns the scalar estimate as well as the
standard error of the mean of the estimate.

### Importance Sampling Cross Validation Criterion (ISCV)

Like the negative WAIC, the ISCV criterion estimates the holdout log-likelihood on future observables using the training data $$(x_i,y_i)_i$$ and a sample
$$(\theta_j)_j$$ from the posterior distribution over parameters.

The ISCV is defined as follows.

$$\textrm{ISCV} := -\frac{1}{n} \sum_{i=1}^n
  \log \frac{1}{m} \sum_{j=1}^m \frac{1}{p(y_i|x_i,\theta_j)}$$

We can estimate the ISCV using the following code:

```python
  import predictive_metrics.posterior_predictive_criteria as ppc

  # logp has shape (n,m), n instances, m ensemble members
  iscv, iscv_sem = ppc.importance_sampling_cross_validation(logp)
```

__References__

* _Alan E. Gelfand_, _Dipak K. Dey_, and _Hong Chang_.
  "Model determination using predictive distributions with implementation via
  sampling-based methods",
  Technical report No. 462, Department of Statistics,
  Stanford university, 1992.
  [(PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.860.3702&rep=rep1&type=pdf)
* _Aki Vehtari_, _Andrew Gelman_, and _Jonah Gabry_.
  "Practical Bayesian model evaluation using leave-one-out cross-validation and
  WAIC",
  arXiv:1507.04544,
  [(PDF)](https://arxiv.org/pdf/1507.04544.pdf)

