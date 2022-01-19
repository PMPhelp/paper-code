"""Creates Mel-filterbank frontends."""

from typing import Callable

import tensorflow as tf


_TensorCallable = Callable[[tf.Tensor], tf.Tensor]
_Initializer = tf.keras.initializers.Initializer


def log_compression(inputs: tf.Tensor,
                    log_offset: float = 1e-5) -> tf.Tensor:
  """Compress an inputs tensor with using a logarithm."""
  return tf.math.log(inputs + log_offset)


class MelFilterbanks(tf.keras.layers.Layer):
  """Computes mel-filterbanks."""

  def __init__(self,
               n_filters: int = 40,
               sample_rate: int = 16000,
               n_fft: int = 512,
               window_len: float = 25.,
               window_stride: float = 10.,
               compression_fn: _TensorCallable = log_compression,
               min_freq: float = 60.0,
               max_freq: float = 7800.0,
               **kwargs):
    """Constructor of a MelFilterbanks frontend.

    Args:
      n_filters: the number of mel_filters.
      sample_rate: sampling rate of input waveforms, in samples.
      n_fft: number of frequency bins of the spectrogram.
      window_len: size of the window, in seconds.
      window_stride: stride of the window, in seconds.
      compression_fn: a callable, the compression function to use.
      min_freq: minimum frequency spanned by mel-filters (in Hz).
      max_freq: maximum frequency spanned by mel-filters (in Hz).
      **kwargs: other arguments passed to the base class, e.g. name.
    """

    super().__init__(**kwargs)

    self._n_filters = n_filters
    self._sample_rate = sample_rate
    self._n_fft = n_fft
    self._window_len = int(sample_rate * window_len // 1000 + 1)
    self._window_stride = int(sample_rate * window_stride // 1000)
    self._compression_fn = compression_fn
    self._min_freq = min_freq
    self._max_freq = max_freq if max_freq else sample_rate / 2.

    self.mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self._n_filters,
        num_spectrogram_bins=self._n_fft // 2 + 1,
        sample_rate=self._sample_rate,
        lower_edge_hertz=self._min_freq,
        upper_edge_hertz=self._max_freq)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Computes mel-filterbanks of a batch of waveforms.

    Args:
      inputs: input audio of shape (batch_size, num_samples).
    Returns:
      Mel-filterbanks of shape (batch_size, time_frames, freq_bins).
    """
    stft = tf.signal.stft(
        inputs,
        frame_length=self._window_len,
        frame_step=self._window_stride,
        fft_length=self._n_fft,
        pad_end=True)

    spectrogram = tf.math.square(tf.math.abs(stft))

    mel_filterbanks = tf.matmul(spectrogram, self.mel_filters)
    mel_filterbanks = self._compression_fn(mel_filterbanks)
    return mel_filterbanks
