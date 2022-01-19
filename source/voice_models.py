"""Neural architectures for audio classification."""

import functools
from typing import Any, Optional, Sequence

import tensorflow as tf


class AudioClassifier(tf.keras.Model):
  """Neural network architecture to train an audio classifier from waveforms."""

  def __init__(self,
               num_outputs: int,
               frontend: Optional[tf.keras.Model] = None,
               encoder: Optional[tf.keras.Model] = None):
    """Initialization.

    Args:
      num_outputs: the number of classes of the classification problem.
      frontend: A keras model that takes a waveform and outputs a time-frequency
        representation.
      encoder: An encoder to turn the time-frequency representation into an
        embedding.
    """
    super().__init__()
    self._frontend = frontend
    self._encoder = encoder
    self._pool = tf.keras.Sequential([
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Flatten(),
    ])
    self._head = tf.keras.layers.Dense(num_outputs, activation=None)

  def call(self, inputs: tf.Tensor, training: bool = True):
    output = inputs
    if self._frontend is not None:
      output = self._frontend(output, training=training)  # pylint: disable=not-callable
      output = tf.expand_dims(output, -1)
    if self._encoder:
      output = self._encoder(output, training=training)
    output = self._pool(output)
    return self._head(output)


class ConvBlock(tf.keras.Sequential):
  """A convolutional block made of two convolutions: time and frequency."""

  def __init__(self,
               num_filters: int,
               activation: Any = 'relu',
               normalization_layer=tf.keras.layers.BatchNormalization,
               dropout: float = 0.0,
               max_pooling: bool = True):
    conv_fn = functools.partial(tf.keras.layers.Conv2D, padding='SAME')
    layers = [
        conv_fn(filters=num_filters, kernel_size=[3, 1]),
        tf.keras.layers.Activation(activation)
    ]
    if normalization_layer is not None:
      layers.append(normalization_layer())
    if dropout > 0.0:
      layers.append(tf.keras.layers.Dropout(dropout))
    layers.extend([
        conv_fn(filters=num_filters, kernel_size=[1, 3]),
        tf.keras.layers.Activation(activation)
    ])
    if normalization_layer is not None:
      layers.append(normalization_layer())
    if dropout > 0.0:
      layers.append(tf.keras.layers.Dropout(dropout))
    if max_pooling:
      layers.append(tf.keras.layers.MaxPool2D())
    super().__init__(layers)


class ConvNet(tf.keras.Sequential):
  """An convolutional encoder made of a sequence of ConvBlocks."""

  def __init__(self, filters: Sequence[int], **kwargs):
    # Uses pooling every second layer
    super().__init__([ConvBlock(depth, max_pooling=(not i or i % 2), **kwargs)
 