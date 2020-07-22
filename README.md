# Uncertainty Metrics
The goal of this library is to provide an easy-to-use interface for both measuring uncertainty across Google and the open-source community.

Machine learning models often produce incorrect (over or under confident) probabilities. In real-world decision making systems, classification models must not only be accurate, but also should indicate when they are likely to be incorrect. For example, one important property is calibration: the idea that a model's predicted probabilities of outcomes reflect true probabilities of those outcomes. Intuitively, for class predictions, calibration means that if a model assigns a class with 90% probability, that class should appear 90% of the time.

## Usage
We access Uncertainty Metrics via `import uncertainty_metrics as um`.

## Support
We support the following calibration metrics:

- Expected Calibration Error [3]
- Root-Mean-Squared Calibration Error [14]
- Static Calibration Error [2]
- Adaptive Calibration Error / Thresholded Adaptive Calibration Error [2]
- General Calibration Error (a space of calibration metrics) [2]
- Class-conditional / Class-conflationary versions of all of the above. [2]
- Bayesian Expected Calibration Error
- Semiparametric Calibration Error

We also support:
- Brier Score
- Brier Decomposition
- Model Uncertainty
- Negative Widely Applicable Information Criterion
- Importance Sampling Cross Validation
- Continuous Ranked Probability Score

## To add a new metric:
1. Add the paper reference to the `References` section below.
2. Add the metric definition to the numpy/ dir for a numpy based metric or to the tensorflow/ dir for a tensorflow based metric.
. Every file should have a subclass of `datasets.base.BaseDataset`, which at a minimum requires implementing a constructor, `_read_examples`, and `_create_process_example_fn`.
3. Add the metric class or function to the corresponding __init__.py file.
4. Add a test that at a minimum implements the metric using 'import uncertainty_metrics as um' and um.*your metric* and checks that the value is in the appropriate range.

## Expected Calibration Error Example
```python
import uncertainty_metrics as um

probabilities = ...
labels = ...
ece = um.ece(labels, probabilities, num_bins=30)
```

## Reliability Diagram Example
```python
import uncertainty_metrics as um

probabilities = ...
labels = ...
diagram = um.reliability_diagram(labels, probabilities)
```

## An example of how to diagnose miscalibration.

Calibration is one of the most important properties of a trained model beyond accuracy. We demonsrate how to calculate calibration measure and diagnose miscalibration with the help of this library. One typical measure of calibration is Expected Calibration Error (ECE)
([Guo et al., 2017](https://arxiv.org/pdf/1706.04599.pdf)). To calculate ECE, we group predictions into M bins (M=15 in our example) according to their confidence, which *in ECE is* the value of the max softmax output, and compute the accuracy in each bin. Let B_m be the set of examples whose predicted confidence falls into the m th interval. The Acc and the Conf of bin B_m is

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Cmathrm%7BAcc%7D(B_m)%3D%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%5Csum_%7Bx_i%20%5Cin%20B_m%7D%20%5Cmathbb%7B1%7D%20(%5Chat%7By_i%7D%20%3D%20y_i)%2C%20%5Cquad%0A%5Cmathrm%7BConf%7D(B_m)%20%3D%20%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%20%5Csum_%7Bx_i%5Cin%20B_m%7D%20%5Chat%7Bp_i%7D%2C%0A%5Cend%7Bequation*%7D">

ECE is defined to be the sum of the absolute value of the difference of Acc and Conf in each bin. Thus, we can see that ECE is designed to measure the alignment between accuracy and confidence. This provides a quantitative way to measure calibration. The better calibration leads to lower ECE.


In this example, we also need to introduce mixup ([Zhang et al., 2017](https://arxiv.org/pdf/1710.09412.pdf)). It is a data-augmentation technique in image classification, which improves both accuracy and calibration in single model. Mixup applies the following only in the ***training***,

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%20%20%20%20%5Clabel%7Beq%3Amixup%7D%0A%20%20%20%20%5Ctilde%7Bx%7D_i%20%3D%20%5Clambda%20x_i%20%2B%20(1-%5Clambda)%20x_j%2C%20%5Cquad%0A%20%20%20%20%5Ctilde%7By%7D_i%20%3D%20%5Clambda%20y_i%20%2B%20(1-%5Clambda)%20y_j.%0A%5Cend%7Balign*%7D">

We focus on the calibration (measured by ECE) of Mixup + BatchEnsemble ([Wen et al., 2020](https://arxiv.org/pdf/2002.06715.pdf)). We first calculate the ECE of some fully ***trained*** models using this library.

```python
import tensorflow as tf
import uncertainty_metrics as um

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


## Stability
There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

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
