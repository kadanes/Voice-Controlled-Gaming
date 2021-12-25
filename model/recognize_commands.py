# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/recognize_commands.py

import collections
import numpy as np


class RecognizeResult(object):

  def __init__(self):
    self._founded_command = "_silence_"
    self._score = 0
    self._is_new_command = False

  @property
  def founded_command(self):
    return self._founded_command

  @founded_command.setter
  def founded_command(self, value):
    self._founded_command = value

  @property
  def score(self):
    return self._score

  @score.setter
  def score(self, value):
    self._score = value

  @property
  def is_new_command(self):
    return self._is_new_command

  @is_new_command.setter
  def is_new_command(self, value):
    self._is_new_command = value


class RecognizeCommands(object):
  def __init__(self, labels, average_window_duration_ms, detection_threshold,
               suppression_ms, minimum_count):
    """Init the RecognizeCommands with parameters used for smoothing."""
    # Configuration
    self._labels = labels
    self._average_window_duration_ms = average_window_duration_ms
    self._detection_threshold = detection_threshold
    self._suppression_ms = suppression_ms
    self._minimum_count = minimum_count
    # Working Variable
    self._previous_results = collections.deque()
    self._label_count = len(labels)
    self._previous_top_label = "_silence_"
    self._previous_top_time = -np.inf

  def process_latest_result(self, latest_results, current_time_ms,
                            recognize_element):

    # print("Scores", end=": " )
    # [print('{:.3f}'.format(float(x)), end=" ") for x in latest_results]
    # print(", Max: {:.3f}".format(max(latest_results)))

    if latest_results.shape[0] != self._label_count:
      raise ValueError("The results for recognition should contain {} "
                       "elements, but there are {} produced".format(
                           self._label_count, latest_results.shape[0]))
    if (self._previous_results.__len__() != 0 and
        current_time_ms < self._previous_results[0][0]):
      raise ValueError("Results must be fed in increasing time order, "
                       "but receive a timestamp of {}, which was earlier "
                       "than the previous one of {}".format(
                           current_time_ms, self._previous_results[0][0]))

    # Add the latest result to the head of the deque.
    self._previous_results.append([current_time_ms, latest_results])

    # Prune any earlier results that are too old for the averaging window.
    time_limit = current_time_ms - self._average_window_duration_ms
    while time_limit > self._previous_results[0][0]:
      self._previous_results.popleft()

    # If there are too few results, the result will be unreliable and bail.
    how_many_results = self._previous_results.__len__()
    earliest_time = self._previous_results[0][0]
    sample_duration = current_time_ms - earliest_time
    if (how_many_results < self._minimum_count or
        sample_duration < self._average_window_duration_ms / 4):
      recognize_element.founded_command = self._previous_top_label
      recognize_element.score = 0.0
      recognize_element.is_new_command = False
      return

    # Calculate the average score across all the results in the window.
    average_scores = np.zeros(self._label_count)
    for item in self._previous_results:
      score = item[1]
      for i in range(score.size):
        average_scores[i] += score[i] / how_many_results

    # Sort the averaged results in descending score order.
    sorted_averaged_index_score = []
    for i in range(self._label_count):
      sorted_averaged_index_score.append([i, average_scores[i]])
    sorted_averaged_index_score = sorted(
        sorted_averaged_index_score, key=lambda p: p[1], reverse=True)

    # Use the information of previous result to get current result
    current_top_index = sorted_averaged_index_score[0][0]
    current_top_label = self._labels[current_top_index]
    current_top_score = sorted_averaged_index_score[0][1]
    time_since_last_top = 0
    if (self._previous_top_label == "_silence_" or
        self._previous_top_time == -np.inf):
      time_since_last_top = np.inf
    else:
      time_since_last_top = current_time_ms - self._previous_top_time
    if (current_top_score > self._detection_threshold and
        current_top_label != self._previous_top_label and
        time_since_last_top > self._suppression_ms):
      self._previous_top_label = current_top_label
      self._previous_top_time = current_time_ms
      recognize_element.is_new_command = True
    else:
      recognize_element.is_new_command = False
    recognize_element.founded_command = current_top_label
    recognize_element.score = current_top_score