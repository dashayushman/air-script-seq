import gzip

import numpy

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

class DataSet(object):

  def __init__(self,
               instances,
               labels,
               reshape=True):
    """Construct a DataSet.
    """
    assert instances.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (instances.shape, labels.shape))
    self._num_examples = instances.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
        instances = instances.reshape(instances.shape[0],
                                instances.shape[1] * instances.shape[2])
    self._instances = instances
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def instances(self):
    return self._instances

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._instances = self._instances[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._instances[start:end], self._labels[start:end]
