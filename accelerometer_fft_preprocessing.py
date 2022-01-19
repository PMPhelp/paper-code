"""Compute FFT based features on limb accelerometer measurements.

1.   Amplitude of each limb (mean vector magnitude)
2.   oscillation amplitidue of each limb (std of vector magitude)

From the above we get 2 features per limb for a total of 8 features.
We could also use frequency of primary oscillation.
"""

def norm_fft(y, T, max_freq=None):
    N = y.shape[0]
    Nf = N // 2 if max_freq is None else int(max_freq * T)
    xf = np.linspace(0.0, 0.5 * N / T, N // 2)
    yf = 2.0 / N * np.fft.fft(y)
    return xf[:Nf], yf[:Nf]


def build_fft_feature_array(input_df):
  """Method to build input feature arrays."""
  output_feats_array=[]
  limb_colnames = ['LA', 'RA', 'LW', 'RW']
  for index, row in input_df.iterrows():
    array_row = []
    for limb_colname in limb_colnames:
      try:
        ft_limb=np.abs(norm_fft(row[limb_colname],45)[1][1])
      except:
        ft_limb =0
        print('exception')
        pass
      array_row.extend([row[limb_colname].mean(), ft_limb])
    output_feats_array.append(array_row)
  return np.array(output_feats_array)