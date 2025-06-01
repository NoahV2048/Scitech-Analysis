import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks

def get_harmonic_peaks(freqs, magnitudes, num_peaks=5, min_prominence=0.01):
    peaks, properties = find_peaks(magnitudes, prominence=min_prominence)
    sorted_indices = np.argsort(magnitudes[peaks])[::-1]
    top_peaks = peaks[sorted_indices[:num_peaks]]
    return [(freqs[p], magnitudes[p]) for p in top_peaks]


def get_df(path):
    df = pd.read_csv(path, delimiter="\t", header=None)
    
    filtered_df = df[df[1] != '[-inf]']
    time = np.array(filtered_df[0], dtype=float)
    amplitude = np.array(filtered_df[1], dtype=float)

    return time, amplitude, filtered_df

def plot_txt(fig, axs, path, index):
    df = pd.read_csv(path, delimiter="\t", header=None)
    
    filtered_df = df[df[1] != '[-inf]']
    time = np.array(filtered_df[0], dtype=float)
    amplitude = np.array(filtered_df[1], dtype=float)

    # axs[index, 0].plot(time, amplitude)

    dt = np.mean(np.diff(time))
    fs = 1.0 / dt

    linear_amplitude = 10 ** (amplitude / 20.0)

    N = len(linear_amplitude)
    fft_vals = np.fft.fft(linear_amplitude)
    freqs = np.fft.fftfreq(N, d=dt)

    fft_offset = 50

    positive_freqs = freqs[fft_offset:N // 2]
    positive_magnitudes = np.abs(fft_vals[fft_offset:N // 2])

    harmonic_peaks = get_harmonic_peaks(positive_freqs, positive_magnitudes, num_peaks=20, min_prominence=0.5)

    # axs[index].plot(time, linear_amplitude)
    axs[index].specgram(linear_amplitude, NFFT=1024, Fs=fs)


directory = r"Audacity-Data\Hybrid-Humbucker\5-100"
dir_files = os.listdir(directory)

plots = len(dir_files)
# plots = 2
fig, axs = plt.subplots(plots, figsize=(12, 20), sharex=True)

for idx, file in enumerate(dir_files):
    plot_txt(fig, axs, f'{directory}\{file}', index=idx)


# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 B Strum.csv', index=0)
# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 N Strum.csv', index=1)

plt.tight_layout()
plt.show()