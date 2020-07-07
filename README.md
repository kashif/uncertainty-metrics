# Metrics for evaluating probability models

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

## Information Criteria

* Widely Applicable Information Criterion (WAIC), type 1 and type 2
* Importance-Sampling Cross Validation (ISCV)

## Proper Scoring Rules

* Categorical Brier score
* Univariate Continuous Ranked Probability Score (CRPS)


## An example on how to diagnose miscalibration.

Calibration is one of the most important properties of a trained model beyond accuracy. We demonsrate how to calculate calibration measure and diagnose miscalibration with the help of this library. One typical measure of calibration is Expected Calibration Error (ECE)
([Guo et al., 2017](https://arxiv.org/pdf/1706.04599.pdf)). To calculate ECE, we group predictions into M bins (M=15 in our example) according to their confidence, which *in ECE is* the value of the max softmax output, and compute the accuracy in each bin. Let B_m be the set of examples whose predicted confidence falls into the m th interval. The Acc and the Conf of bin B_m is

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%5Cmathrm%7BAcc%7D(B_m)%3D%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%5Csum_%7Bx_i%20%5Cin%20B_m%7D%20%5Cmathbb%7B1%7D%20(%5Chat%7By_i%7D%20%3D%20y_i)%2C%20%5Cquad%0A%5Cmathrm%7BConf%7D(B_m)%20%3D%20%5Cfrac%7B1%7D%7B%7CB_m%7C%7D%20%5Csum_%7Bx_i%5Cin%20B_m%7D%20%5Chat%7Bp_i%7D%2C%0A%5Cend%7Bequation*%7D">

ECE is defined to be the sum of the absolute value of the difference of Acc and Conf in each bin. Thus, we can see that ECE is designed to measure the alignment between accuracy and confidence. This provides a quantitative way to measure calibration. The better calibration leads to lower ECE.


In this example, we also need to introduce mixup ([Zhang et al., 2017](https://arxiv.org/pdf/1710.09412.pdf)). It is a data-augmentation technique in image classification, which improves both accuracy and calibration in single model. Mixup applies the following only in the ***training***,

  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%20%20%20%20%5Clabel%7Beq%3Amixup%7D%0A%20%20%20%20%5Ctilde%7Bx%7D_i%20%3D%20%5Clambda%20x_i%20%2B%20(1-%5Clambda)%20x_j%2C%20%5Cquad%0A%20%20%20%20%5Ctilde%7By%7D_i%20%3D%20%5Clambda%20y_i%20%2B%20(1-%5Clambda)%20y_j.%0A%5Cend%7Balign*%7D">

We focus on the calibration (measured by ECE) of Mixup + BatchEnsemble ([Wen et al., 2020](https://arxiv.org/pdf/2002.06715.pdf)). We first calculate the ECE of some fully ***trained*** models using this library.

```python
import tensorflow as tf
from uncertainty_metrics import general_calibration_error

# Load and preprocess a dataset. Also load the model.
test_images, test_labels = ...
model = ...

# Obtain predictive probabilities.
probs = model(test_images, training=False) # probs is of shape [4, testset_size, num_classes] if the model is an ensemble of 4 individual models.
ensemble_probs = tf.reduce_mean(model, axis=0)

# Calculate individual calibration error.
individual_eces = []
for i in range(ensemble_size):
  individual_eces.append(general_calibration_error.ece(probs[i], labels, num_bins=15))
  
ensemble_ece = general_calibration_error.ece(ensemble_probs, labels, num_bins=15)
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
ensemble_metric = general_calibration_error.GeneralCalibrationError(
    num_bins=15,
    binning_scheme='even',
    class_conditional=False,
    max_prob=True,
    norm='l1')
ensemble_metric.update_state(ensemble_probs, labels)

individual_metric = general_calibration_error.GeneralCalibrationError(
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
