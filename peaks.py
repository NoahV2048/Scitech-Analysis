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
    print("Average Signal Power", signal_power)

    noise_power = np.mean(noise**2)
    print("Average Noise Power", noise_power)

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

def plot_txt(fig, axs, path: str, index, output="Output-Data"):
    df = pd.read_csv(path, delimiter="\t", header=None)
    
    filtered_df = df[df[1] != '[-inf]']
    time = np.array(filtered_df[0], dtype=float)
    amplitude = np.array(filtered_df[1], dtype=float)

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

    harmonic_peaks_sorted = sorted(harmonic_peaks, key=lambda x: x[0])  # Sort by frequency

    output_df = pd.DataFrame(columns=["Frequency (Hz)", "Magnitude"])

    for idx, (freq, mag) in enumerate(harmonic_peaks_sorted):
        output_df = output_df._append({
            "Frequency (Hz)": freq,
            "Magnitude": mag
        }, ignore_index=True)
        print(mag)

    np.savetxt(f'{output}\{path.split('\\')[-1]}', output_df.to_numpy(), delimiter='\t')

directory = f"Audacity-Data\\Hybrid-Humbucker\\100-100"
dir_files = os.listdir(directory)

plots = len(dir_files)
# plots = 2
fig, axs = plt.subplots(plots, figsize=(12, 20), sharex=True)

for idx, file in enumerate(dir_files):
    plot_txt(fig, axs, f'{directory}\{file}', index=idx, output="Output-Data")

exit(0)

for i in range(0, 50, 5):
    directory = f"Audacity-Data\\Hybrid-Humbucker\\{i}-100"
    dir_files = os.listdir(directory)

    plots = len(dir_files)
    # plots = 2
    fig, axs = plt.subplots(plots, figsize=(12, 20), sharex=True)

    for idx, file in enumerate(dir_files):
        plot_txt(fig, axs, f'{directory}\{file}', index=idx, output="Output-Data")

for i in range(50, 100, 10):
    directory = f"Audacity-Data\\Hybrid-Humbucker\\{i}-100"
    dir_files = os.listdir(directory)

    plots = len(dir_files)
    # plots = 2
    fig, axs = plt.subplots(plots, figsize=(12, 20), sharex=True)

    for idx, file in enumerate(dir_files):
        plot_txt(fig, axs, f'{directory}\{file}', index=idx, output="Output-Data")


# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 B Strum.csv', index=0)
# plot_txt(fig, axs, r'Audacity-Data\Hybrid-Humbucker\10-100\Hybrid 10-100 N Strum.csv', index=1)

# plt.tight_layout()
# plt.show()