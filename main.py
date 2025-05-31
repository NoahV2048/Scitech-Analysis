import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks

def compute_snr(signal, noise):
    """
    Computes SNR in decibels from two segments: signal and noise (in linear amplitude).

    Args:
        signal (np.ndarray): Signal segment (linear scale).
        noise (np.ndarray): Noise segment (linear scale).

    Returns:
        float: SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def compute_snr_from_files(signalfile, noisefile):
    signal_df = pd.read_csv(signalfile, delimiter="\t", header=None)
    filtered_signal_df = np.array(signal_df[signal_df[1] != '[-inf]'], dtype=float)
    noise_df = pd.read_csv(noisefile, delimiter="\t", header=None)
    filtered_noise_df = np.array(noise_df[noise_df[0] != '[-inf]'], dtype=float)

    print(compute_snr(10 ** (filtered_signal_df[1] / 20.0), 10 ** (filtered_noise_df[1] / 20.0)))

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

    axs[index].plot(positive_freqs, positive_magnitudes)
    for freq, mag in harmonic_peaks:
        axs[index].plot(freq, mag, 'ro')
        axs[index].annotate(f"{freq:.1f} Hz", (freq, mag), textcoords="offset points", xytext=(0, 10), ha='center')

    # print(harmonic_peaks)

    axs[index].set_xscale('log')


directory = r"Audacity-Data\Hybrid-Humbucker\5-100"
dir_files = os.listdir(directory)

plots = len(dir_files)
# plots = 2
fig, axs = plt.subplots(plots, figsize=(12, 20), sharex=True)

for idx, file in enumerate(dir_files):
    plot_txt(fig, axs, f'{directory}\{file}', index=idx)

compute_snr_from_files(
    r"Audacity-Data\Hybrid-Humbucker\5-100\Hybrid 5-100 N Strum.csv", 
    r"Audacity-Data\Hybrid-Humbucker\Noise\Hybrid 0-100 Noise.csv"
)

compute_snr_from_files(
    r"Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 N Strum.csv", 
    r"Audacity-Data\Hybrid-Humbucker\Noise\Hybrid 10-100 Noise.csv"
)

# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 B Strum.csv', index=0)
# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 N Strum.csv', index=1)

plt.tight_layout()
plt.show()
