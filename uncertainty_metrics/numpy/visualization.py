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

# Lint as: python3
"""Reliability Diagrams and other calibration visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np


def one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def mean(inputs):
  """Be able to take the mean of an empty array without hitting NANs."""
  # pylint disable necessary for numpy and pandas
  if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
    return 0
  else:
    return np.mean(inputs)


def binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1-p, p] for p in probs])


def verify_probability_shapes(probs):
  """Verify shapes of probs vectors and possibly stack 1D probs into 2D."""
  if probs.ndim == 2:
    num_classes = probs.shape[1]
    if num_classes == 1:
      probs = probs[:, 0]
      probs = binary_converter(probs)
      num_classes = 2
  elif probs.ndim == 1:
    # Cover binary case
    probs = binary_converter(probs)
    num_classes = 2
  else:
    raise ValueError('Probs must have 1 or 2 dimensions.')
  return probs, num_classes


def to_image(fig):
  """Create image from plot."""
  fig.tight_layout(pad=1)
  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  return image_from_plot


def plot_diagram(probs, labels, y_axis='accuracy'):
  """Helper function operating on 1D representation of probs and labels."""
  probs_labels = [(prob, labels[i]) for i, prob in enumerate(probs)]
  probs_labels = np.array(sorted(probs_labels, key=lambda x: x[0]))
  window_len = int(len(labels)/100.)
  calibration_errors = []
  confidences = []
  accuracies = []
  # More interesting than length of the window (which is specific to this
  # window) is average distance between datapoints. This normalizes by dividing
  # by the window length.
  distances = []
  for i in range(len(probs_labels)-window_len):
    distances.append((
        probs_labels[i+window_len, 0] - probs_labels[i, 0])/float(window_len))
    # It's pretty sketchy to look for the 100 datapoints around this one.
    # They could be anywhere in the probability simplex. This introduces bias.
    mean_confidences = mean(probs_labels[i:i + window_len, 0])
    confidences.append(mean_confidences)
    class_accuracies = mean(probs_labels[i:i + window_len, 1])
    accuracies.append(class_accuracies)
    calibration_error = class_accuracies-mean_confidences
    calibration_errors.append(calibration_error)

  if y_axis == 'accuracy':
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    xbins = [i/10. for i in range(11)]
    ax.plot(confidences, accuracies, color='green')
    ax.plot(xbins, xbins, color='orange')
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Model Accuracy')
  elif y_axis == 'error':
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    xbins = [i/10. for i in range(11)]
    ax.plot(confidences, calibration_errors, color='green')
    ax.plot(xbins, xbins, color='orange')
    ax.set_xlabel('Model Confidence')
    ax.set_ylabel('Model Calibration Error')
  ax.set_title('Reliability Diagram', fontsize=20)
  return fig


def reliability_diagram(labels,
                        probs,
                        class_conditional=False,
                        y_axis='accuracy',
                        img=False):
  """Reliability Diagram plotting confidence against accuracy.

  Note that this reliability diagram is created by looking at the calibration of
    the set of datapoints that surround each datapoint, not through mutually
    exclusive bins.

  Args:
    labels: label vector.
    probs: probability matrix out of a softmax.
    class_conditional: whether to visualize every class independently, or
      conflate classes.
    y_axis: takes 'accuracy or 'error'. Set y_axis to 'error' to graph the
      calibration error (confidence - accuracy) against the accuracy instead.
    img: return as image rather than as a plot.
  Returns:
    fig: matplotlib.pyplot figure.
  """
  probs = np.array(probs)
  labels = np.array(labels)
  probs, _ = verify_probability_shapes(probs)

  labels_matrix = one_hot_encode(labels, probs.shape[1])
  if class_conditional:
    for class_index in range(probs.shape[1]):
      if img:
        return to_image(plot_diagram(
            probs[:, class_index], labels_matrix[:, class_index], y_axis))
      else:
        return plot_diagram(
            probs[:, class_index], labels_matrix[:, class_index], y_axis)
  else:
    if img:
      return to_image(
          plot_diagram(probs.flatten(), labels_matrix.flatten(), y_axis))
    else:
      return plot_diagram(probs.flatten(), labels_matrix.flatten(), y_axis)


def plot_confidence_vs_accuracy_diagram(in_probs,
                                        in_labels,
                                        out_probs,
                                        num_thresholds=20,
                                        ax=None,
                                        **plot_kwargs):
  """Accuracy vs confidence diagrams plotting confidence against accuracy.

  Accuracy vs confidence diagrams are plotted to measure a model's OOD
  performance by evaluating the accuracy of detecting OOD samples only on cases
  where the model's confidence is above a threshold. We define confidence as the
  maximum of the class probabilities, and consider all cases where the
  confidence is above a threshold. For in-distribution cases, we evaluate the
  label prediction accuracy, and for OOD cases, we consider all predictions
  incorrect. For further reference, refer to Section 3.6 of
  https://arxiv.org/abs/1612.01474.

  Args:
    in_probs: probability matrix for in-distribution points.
    in_labels: label vector for in-distribution points.
    out_probs: probability matrix out of a softmax for out_of_distribution
      points.
    num_thresholds: (int) number of thresholds in [0., 0.99] to evaluate
      accuracy of OOD detection on.
    ax: matplotlib.pyplot Axes object to plot the figure on. If None, the latest
      figure used is grabbed using `plt.gca()`.
    **plot_kwargs: Optional plotting arguments to pass into `ax.plot()`.
  Returns:
    fig: matplotlib.pyplot Axes object.
  """
  in_probs = np.array(in_probs)
  out_probs = np.array(out_probs)
  in_labels = np.array(in_labels)
  in_probs, _ = verify_probability_shapes(in_probs)
  out_probs, _ = verify_probability_shapes(out_probs)

  thresholds = np.linspace(0.0, 0.99, num=num_thresholds)
  in_confidences = np.max(in_probs, axis=-1)
  out_confidences = np.max(out_probs, axis=-1)
  in_predictions = np.argmax(in_probs, axis=-1)
  in_predictions_correct = (in_predictions == in_labels)
  accuracies = []
  plotting_thresholds = []
  for i in thresholds:
    # Check that there are points above the threshold
    if np.sum(in_confidences >= i) + np.sum(out_confidences >= i) > 0:
      in_above_threshold_correct = in_predictions_correct[in_confidences >= i]
      out_above_threshold_predictions = np.zeros(np.sum(out_confidences >= i))

      all_predictions = np.hstack([in_above_threshold_correct.astype('int'),
                                   out_above_threshold_predictions])
      accuracies.append(np.mean(all_predictions))
      plotting_thresholds.append(i)
    else:
      break

  if ax is None:
    ax = plt.gca()
  ax.plot(plotting_thresholds, accuracies, **plot_kwargs)
  ax.set_xlabel(r'Confidence Threshold $\tau$')
  ax.set_ylabel(r'Accuracy on examples $p(y|x) \geq \tau$')
  ax.set_title('Accuracy vs Confidence Diagram', fontsize=20)
  ax.set_ylim([0.0, 1.0])
  ax.set_xlim([0.0, 1.0])
  return ax
