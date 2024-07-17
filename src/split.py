import numpy as np
from scipy.ndimage import zoom

zoom_factors = (500 / 1250, 500 / 1250, 500 / 1250)

# Perform the downsampling

# def resample_signal(signal):
#     return resample(signal, 500)

wav_data = np.load("/home/stan/data/pinn/wave_data.npy")
downsampled_data = zoom(wav_data, zoom_factors)
print("Original shape:", wav_data.shape)

# wav_data = np.apply_along_axis(resample_signal, 0, wav_data)
# wav_data = np.apply_along_axis(resample_signal, 1, wav_data)
# wav_data = np.apply_along_axis(resample_signal, 2, wav_data)

print("Resampled shape:", downsampled_data.shape)
np.save("downsampled_data.npy", downsampled_data)
