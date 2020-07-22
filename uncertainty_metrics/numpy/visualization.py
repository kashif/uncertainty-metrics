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


def reliability_diagram(probs,
                        labels,
                        class_conditional=False,
                        y_axis='accuracy',
                        img=False):
  """Reliability Diagram plotting confidence against accuracy.

  Note that this reliability diagram is created by looking at the calibration of
    the set of datapoints that surround each datapoint, not through mutually
    exclusive bins.

  Args:
    probs: probability matrix out of a softmax.
    labels: label vector.
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
  labels_matrix = one_hot_encode(labels)
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
