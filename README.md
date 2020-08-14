# Uncertainty Metrics
The goal of this library is to provide an easy-to-use interface for measuring uncertainty across Google and the open-source community.

Machine learning models often produce incorrect (over or under confident) probabilities. In real-world decision making systems, classification models must not only be accurate, but also should indicate when they are likely to be incorrect. For example, one important property is calibration: the idea that a model's predicted probabilities of outcomes reflect true probabilities of those outcomes. Intuitively, for class predictions, calibration means that if a model assigns a class with 90% probability, that class should appear 90% of the time.

## Installation

```sh
pip install uncertainty_metrics
```

To install the latest development version, run

```sh
pip install "git+https://github.com/google/uncertainty_metrics.git#egg=uncertainty_metrics"
```

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

## Getting Started

Here are some examples to get you started.

__Expected Calibration Error.__

```python
import uncertainty_metrics.numpy as um

probabilities = ...
labels = ...
ece = um.ece(labels, probabilities, num_bins=30)
```

__Reliability Diagram.__

```python
import uncertainty_metrics.numpy as um

probabilities = ...
labels = ...
diagram = um.reliability_diagram(labels, probabilities)
```

__Brier Score.__

```python
import uncertainty_metrics as um

tf_probabilities = ...
labels = ...
bs = um.brier_score(labels=labels, probabilities=tf_probabilities)
```

__How to diagnose miscalibration.__ Calibration is one of the most important properties of a trained model beyond accuracy. We demonsrate how to calculate calibration measure and diagnose miscalibration with the help of this library. One typical measure of calibration is Expected Calibration Error (ECE)
([Guo et al., 2017](https://arxiv.org/pdf/1706.04599.pdf)). To calculate ECE, we group predictions into M bins (M=15 in our example) according to their confidence, which *in ECE is* the value of the max softmax output, and compute the accuracy in each bin. Let B_m be the set of examples whose predicted confidence falls into the m th interval. The Acc and the Conf of bin B_m is

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Cmathrm%7BAcc%7D(B_m)%3D%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%5Csum_%7Bx_i%20%5Cin%20B_m%7D%20%5Cmathbb%7B1%7D%20(%5Chat%7By_i%7D%20%3D%20y_i)%2C%20%5Cquad%0A%5Cmathrm%7BConf%7D(B_m)%20%3D%20%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%20%5Csum_%7Bx_i%5Cin%20B_m%7D%20%5Chat%7Bp_i%7D%2C%0A%5Cend%7Bequation*%7D">

ECE is defined to be the sum of the absolute value of the difference of Acc and Conf in each bin. Thus, we can see that ECE is designed to measure the alignment between accuracy and confidence. This provides a quantitative way to measure calibration. The better calibration leads to lower ECE.


In this example, we also need to introduce mixup ([Zhang et al., 2017](https://arxiv.org/pdf/1710.09412.pdf)). It is a data-augmentation technique in image classification, which improves both accuracy and calibration in single model. Mixup applies the following only in the ***training***,

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%20%20%20%20%5Clabel%7Beq%3Amixup%7D%0A%20%20%20%20%5Ctilde%7Bx%7D_i%20%3D%20%5Clambda%20x_i%20%2B%20(1-%5Clambda)%20x_j%2C%20%5Cquad%0A%20%20%20%20%5Ctilde%7By%7D_i%20%3D%20%5Clambda%20y_i%20%2B%20(1-%5Clambda)%20y_j.%0A%5Cend%7Balign*%7D">

We focus on the calibration (measured by ECE) of Mixup + BatchEnsemble ([Wen et al., 2020](https://arxiv.org/pdf/2002.06715.pdf)). We first calculate the ECE of some fully ***trained*** models using this library.

```python
import tensorflow as tf
import uncertainty_metrics.numpy as um

# Load and preprocess a dataset. Also load the model.
test_images, test_labels = ...
model = ...

# Obtain predictive probabilities.
probs = model(test_images, training=False) # probs is of shape [4, testset_size, num_classes] if the model is an ensemble of 4 individual models.
ensemble_probs = tf.reduce_mean(model, axis=0)

# Calculate individual calibration error.
individual_eces = []
for i in range(ensemble_size):
  individual_eces.append(um.ece(probs[i], labels, num_bins=15))
  
ensemble_ece = um.ece(ensemble_probs, labels, num_bins=15)
```

We collect the ECE in the following table.

| Method/Metric |    | CIFAR-10 |      | CIFAR-100 |      |
|:-------------:|:--:|:--------:|:----:|:---------:|:----:|
|               |    |    Acc   |  ECE |    Acc    |  ECE |
| BatchEnsemble | In |   95.88  | 2.3% |   80.64   | 8.7% |
|               | En |   96.22  | 1.8% |   81.85   | 2.8% |
|  Mixup0.2 BE  | In |   96.43  | 0.8% |   81.44   | 1.5% |
|               | En |   96.75  | 1.5% |   82.79   | 3.9% |
|   Mixup1 BE   | In |   96.67  | 5.5% |   81.32   | 6.6% |
|               | En |   96.98  | 6.4% |   83.12   | 9.7% |

In the above table, ***In*** stands for individual model; ***En*** stands for ensemble models. ***Mixup0.2*** stands for small mixup augmentation while ***mixup1*** stands for strong mixup augmentation. Ensemble typically improves both accuracy and calibration, but this does not apply to mixup. Scalars obsure useful information, so we try to understand more insights by examining the per-bin result.

```python
ensemble_metric = um.GeneralCalibrationError(
    num_bins=15,
    binning_scheme='even',
    class_conditional=False,
    max_prob=True,
    norm='l1')
ensemble_metric.update_state(ensemble_probs, labels)

individual_metric = um.GeneralCalibrationError(
    num_bins=15,
    binning_scheme='even',
    class_conditional=False,
    max_prob=True,
    norm='l1')
for i in range(4)
  individual_metric.update_state(probs[i], labels)
  
ensemble_reliability = ensemble_metric.accuracies - ensemble_metric.confidences
individual_reliability = (
    individual_metric.accuracies - individual_metric.confidences)
```

Now we can plot the reliability diagram which demonstrates more details of calibration. The backbone model in the following figure is BatchEnsemble with ensemble size 4. The plot has 6 lines: we trained three independent BatchEnsemble models with large, small, and no Mixup; and for each model, we compute the calibration of both ensemble and individual predictions. The plot shows that only Mixup models have positive (Acc - Conf) values on the test set, which suggests that Mixup encourages underconfidence. Mixup ensemble's positive value is also greater than Mixup individual's. This suggests that Mixup ensembles compound in encouraging underconfidence, leading to worse calibration than when not ensembling. Therefore, we successfully find the reason why Mixup+Ensemble leads to worse calibration, by leveraging this library.

<img src="https://drive.google.com/uc?export=view&id=1M-raNJyzsNBHhGuPoVfSmUtrSKOLPx3U" width="750"/>

## Background & API

Uncertainty Metrics provides several types of measures of probabilistic error:

- Calibration error
- Proper scoring rules
- Information critera
- Diversity
- AUC/Rejection
- Visualization tools

We outline each type below.

### Calibration Error

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

We support the following calibration metrics:

- Expected Calibration Error [3]
- Root-Mean-Squared Calibration Error [14]
- Static Calibration Error [2]
- Adaptive Calibration Error / Thresholded Adaptive Calibration Error [2]
- General Calibration Error (a space of calibration metrics) [2]
- Class-conditional / Class-conflationary versions of all of the above. [2]
- Bayesian Expected Calibration Error
- Semiparametric Calibration Error

We describe examples below.

__Example: Expected Calibration Error.__
The expected calibration error (ECE) is a scalar summary statistic between zero
(perfect calibration) and one (fully miscalibrated).
It is widely reported in the literature as a standard summary statistic
for classification.

ECE works by binning the probability of the _decision label_ into a
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
features, labels = ...  # get from minibatch
probs = model(features)
ece = um.ece(probs=probs, labels=labels, num_bins=10)
```

__Example: Bayesian Expected Calibration Error.__
ECE is a scalar summary statistic of miscalibration
evaluated on a finite sample of validation data.  Resulting in a single scalar,
the sampling variation due to the limited amount of validation data is hidden,
and this can result in significant over- or under-estimation of the ECE as well
as wrongly concluding significant differences in ECE between multiple models.

To address these issues, a Bayesian estimator of the ECE can be used.  The
resulting estimate is not a single scalar summary but instead a probability
distribution over possible ECE values.

![drawing](https://docs.google.com/drawings/d/1w4GFeDRi0aIYgcalPy1QFguBe7Je1C6w2_Fx3VBMXVE/export/png)

The generative model is given by the following generative mechanism.

$$q \sim \textrm{Dirichlet}(\alpha \, \mathbb{1}_{2\times M}),$$

The distribution $$q$$ is over $$2 \times M$$ outcomes, where the first $$M$$
outcomes mean $$z_i = 0$$ and the second set of $$M$$ outcomes mean $$z_i = 1$$.
Thus,

$$(z_i,b_i) \sim \textrm{Categorical}(q), \quad i=1,\dots,n,$$

The per-bin means are generated by a
[truncated Normal distribution](https://en.wikipedia.org/wiki/Truncated_normal_distribution),

$$p_i \sim \textrm{TruncatedNormal}(\mu_{b_i}, \sigma^2, I_{b_i}),
\quad i=1,\dots,n.$$

For inference, we are given a list of $$(\hat{y}_i,y_i,p_i)$$ triples, where:

*   $$\hat{y}_i$$ is the decision label of the prediction system, typically
    being the $$\textrm{argmax}$$ over the predicted probabilities;
*   $$y_i$$ is the true observed label; and
*   $$p_i$$ is the probability $$p(\hat{y}_i|x_i)$$ provided by the prediction
    system.

Each probability $$p_i$$ can be uniquely assigned to a bin $$b_i = b(p_i)$$, and
each pair $$(\hat{y}_i,y)$$ can be uniquely assigned to $$z_i =
\mathbf{1}_{\{\hat{y}_i = y\}}$$.

The posterior $$p(\alpha,\mu|Z,B,P)$$ factorizes as

$$p(q,\mu|Z,B,P) = p(q|Z,B) \, p(\mu|P,B).$$

The first factor $$p(q|Z,B)$$ is a Dirichlet with analytic solution in the
Dirichlet-Multinomial model. The second factor has an
[analytic posterior in the Normal model](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf),
and the truncation does not affect this because it can be seen as a product with
an indicator function over the respective probability interval $$I_b \subset
[0,1]$$.

Given the posterior $$p(q,\mu|Z,B,P)$$ we draw samples $$(q,p)$$ and compute the
analytic ECE formula for each sample,

$$ECE(q,p) := \sum_{m=1}^M q_{1,m} |q_{1,m} - p_m|.$$

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
ece_samples = um.bayesian_expected_calibration_error(
    10, logits=logits, labels_true=labels_true)

ece_quantiles = tfp.stats.percentile(ece_samples, [10,50,90])
```

The above code also includes an example of using the samples to infer
10%/50%/90% quantiles of the distribution of possible ECE values.

### Proper Scoring Rules

_Proper scoring rules_ are loss functions for probabilistic predictions.
Formally, a proper scoring rule is a function which assign a numerical score
$$S(P,y)$$ to a predicted distribution $$P$$ and a realized value $$y$$.
Assuming that data instances are generated according to an unknown distribution,
$$(x_i,y_i) \sim Q$$, we would evaluate $$S(P(y|x_i),y_i)$$.

__Example: Brier score.__
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
per_example_loss = um.brier_score(labels=target_labels, logits=logits)
loss = tf.reduce_mean(per_example_loss)
```

The Brier score penalizes low-probability predictions which do occur less than
the cross-entropy loss.

__Example: Brier score's decomposition.__
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
uncert, resol, reliab = um.brier_decomposition(labels=labels, logits=logits)
```

In particular, the reliability (`reliab` in the above line) is a measure of
calibration error, with a value between 0 and 2, where zero means perfect
calibration.

__Example: Continuous Ranked Probability Score (CRPS).__
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

For a regression model which predicts Normal distributions with mean `means` and
standard deviation `stddevs`, we can compute CRPS as follows:

```python
squared_errors = tf.square(target_labels - pred_means)
per_example_crps = um.crps_normal_score(
    labels=target_labels,
    means=pred_mean,
    stddevs=pred_stddevs)
```

For non-Normal models, as long as we can sample predictions, we can construct a
Tensor `predictive_samples` of size `(ninstances, npredictive_samples)` and
evaluate the Monte Carlo CRPS against the true targets `target_labels` using the
following code,

```python
per_example_crps = um.crps_score(
    labels=target_labels,
    predictive_samples=predictive_samples)
```

### Information Criteria

_Information criteria_ are used after or during model training to estimate the predictive performance on future holdout data.  They can be useful for selecting among multiple possible models or to perform hyperparameter optimization.  There are also strong connections between [cross validation estimates](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) and some information criteria.

We estimate information criteria using log-likelihood values on training samples.
In particular, for both the WAIC and the ISCV criteria we assume that we have an ensemble of models with equal weights, such that the average predictive distribution over the ensemble is a good approximation to the true Bayesian posterior predictive distribution.

For a set of $$n$$ training instances and $$m$$ ensemble models we will use a
Tensor of shape $$(n,m)$$, such that the `[i,j]` element contains
$$\log p(y_i | x_i, \theta_j)$$, where $$\theta_j$$ represents the `j`'th ensemble element.  For example, the $$\theta_j$$ values can be obtained by MCMC sampling or as samples from a variational posterior distribution over parameters.

If the $$\theta$$ samples closely approximate the true posterior over parameters, then the WAIC and ISCV information criteria have strong theoretical properties in that they consistently predict the generalization loss with low variance.

__Example: Mutual information.__
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

Model uncertainty measures the mutual information between the categorical output
$$y$$ and the model parameters $$\theta$$. It is the difference of the total
uncertainty and expected data uncertainty between $$y$$ and $$\theta$$. The
implementation uses the logits to calculate model uncertainty. It returns
knowledge uncertainty as well as total uncertainty and expected data
uncertainty. An example code is given below:

```python
logits = model(validation_data)
model_uncert, total_uncert, avg_data_uncert = um.model_uncertainty(logits)
```

__Example: Widely/Watanabe Applicable Information Criterion (WAIC).__
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
# logp has shape (n,m), n instances, m ensemble members
neg_waic, neg_waic_sem = um.negative_waic(logp, waic_type="waic1")
```

You can select the type of nWAIC to estimate using `waic_type="waic1"` or
`waic_type="waic2"`.  The method returns the scalar estimate as well as the
standard error of the mean of the estimate.

__Example: Importance Sampling Cross Validation Criterion (ISCV).__
Like the negative WAIC, the ISCV criterion estimates the holdout log-likelihood on future observables using the training data $$(x_i,y_i)_i$$ and a sample
$$(\theta_j)_j$$ from the posterior distribution over parameters.

$$\textrm{ISCV} := -\frac{1}{n} \sum_{i=1}^n
  \log \frac{1}{m} \sum_{j=1}^m \frac{1}{p(y_i|x_i,\theta_j)}$$

We can estimate the ISCV using the following code:

```python
# logp has shape (n,m), n instances, m ensemble members
iscv, iscv_sem = um.importance_sampling_cross_validation(logp)
```

## To add a new metric

1. Add the paper reference to the `References` section below.
2. Add the metric definition to the numpy/ dir for a numpy based metric or to the tensorflow/ dir for a tensorflow based metric.s
3. Add the metric class or function to the corresponding __init__.py file.
4. Add a test that at a minimum implements the metric using 'import uncertainty_metrics as um' and um.*your metric* and checks that the value is in the appropriate range.

## References

[1] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017, August). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning. Paper Link.

[2] Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D. (2019). Measuring Calibration in Deep Learning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (pp. 38-41). Paper Link.

[3] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht. "Obtaining well calibrated probabilities using bayesian binning." Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015. Paper Link.

[4] Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht. "Binary classifier calibration using a Bayesian non-parametric approach." Proceedings of the 2015 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2015. Paper Link.

[5] J. Platt. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. Advances in Large Margin Classifiers, 10(3):61–74, 1999. Paper Link.

[6] Kumar, A., Liang, P. S., & Ma, T. (2019). Verified uncertainty calibration. In Advances in Neural Information Processing Systems (pp. 3787-3798). Paper Link.

[7] Kumar, A., Sarawagi, S., & Jain, U. (2018, July). Trainable calibration measures for neural networks from kernel mean embeddings. In International Conference on Machine Learning (pp. 2805-2814). Paper Link.
[8] Calibrating Neural Networks Documentation. Link.

[9] Zadrozny, Bianca, and Charles Elkan. "Transforming classifier scores into accurate multiclass probability estimates." Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining. 2002. Paper Link.

[10] Müller, Rafael, Simon Kornblith, and Geoffrey E. Hinton. "When does label smoothing help?." Advances in Neural Information Processing Systems. 2019. Paper Link.

[11] Pereyra, Gabriel, et al. "Regularizing neural networks by penalizing confident output distributions." arXiv preprint arXiv:1701.06548 (2017). Paper Link.

[12] Lakshminarayanan, B., Pritzel, A., and Blundell, C. Simple and scalable predictive uncertainty estimation using deep ensembles. In NIPS, pp. 6405–6416. 2017. Paper Link.

[13] Louizos, C. and Welling, M. Multiplicative normalizing flows for variational Bayesian neural networks. In ICML, volume 70, pp. 2218–2227, 2017. Paper Link.

[14] Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., & Lakshminarayanan, B. (2019). AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. Paper Link.
    
[15] _Jochen Brocker_, "Reliability, sufficiency, and the decomposition of
proper scores", Quarterly Journal of the Royal Meteorological Society, 2009.
[(PDF)](https://rmets.onlinelibrary.wiley.com/doi/pdf/10.1002/qj.456)

[16]   _Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez, and
Steffen Udluft_, "Decomposition of uncertainty for active learning and reliable
reinforcement learning in stochastic systems", stat 1050, 2017.
[(PDF)](https://proceedings.mlr.press/v80/depeweg18a/depeweg18a.pdf)

[17] _Alan E. Gelfand_, _Dipak K. Dey_, and _Hong Chang_. "Model determination
using predictive distributions with implementation via sampling-based methods",
Technical report No. 462, Department of Statistics, Stanford university, 1992.
[(PDF)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.860.3702&rep=re
p1&type=pdf)

[18] _Tilmann Gneiting_ and _Adrian E. Raftery_, "Strictly Proper Scoring Rules,
Prediction, and Estimation", Journal of the American Statistical Association
(JASA), 2007.
[(PDF)](https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pd
f)

[19]   _Andrey Malinin, Bruno Mlodozeniec and Mark Gales_, "Ensemble
Distribution Distillation.", arXiv:1905.00076, 2019.
[(PDF)](https://arxiv.org/pdf/1905.00076.pdf)

[20] _Aki Vehtari_, _Andrew Gelman_, and _Jonah Gabry_. "Practical Bayesian
model evaluation using leave-one-out cross-validation and WAIC",
arXiv:1507.04544, [(PDF)](https://arxiv.org/pdf/1507.04544.pdf)

[21] _Sumio Watanabe_, "Mathematical Theory of Bayesian Statistics", CRC Press,
2018.
