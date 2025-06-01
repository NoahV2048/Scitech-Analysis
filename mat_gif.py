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

def plot_txt(fig, axs, path, r, c, label, coils=0):
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
    positive_magnitudes = np.abs(fft_vals[fft_offset:N // 2])\
    
    
    axs[r, c].clear()

    axs[r, c].plot(positive_freqs, positive_magnitudes)

    # harmonic_peaks = get_harmonic_peaks(positive_freqs, positive_magnitudes, num_peaks=20, min_prominence=0.5)

    # axs[index].plot(positive_freqs, positive_magnitudes)
    # for freq, mag in harmonic_peaks:
    #     print(mag)
    #     axs[index].plot(freq, mag, 'ro')
    #     axs[index].annotate(f"{mag:.1f}", (freq, mag), textcoords="offset points", xytext=(0, 10), ha='center')

    # print(harmonic_peaks)


    axs[r, c].set_xscale('log')

    axs[r, c].set_xscale('log')
    axs[r, c].set_ylabel(f'db ({label})')

    # Only set x-axis label on bottom-most plot
    if r == 3:
        axs[r, c].set_xlabel('Frequency (Hz)')

    # if index == 0:
    #     axs[index].set_title(f"Coils={coils}")


directory = r"Audacity-Data\Hybrid-Humbucker\30-100"
dir_files = os.listdir(directory)

# plots = len(dir_files)
plots = 6
fig, axs = plt.subplots(3, 2, sharex=True)

# for idx, file in enumerate(dir_files):
#     plot_txt(fig, axs, f'{directory}\{file}', index=idx)

for coils in range(0, 50, 5):
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E2.csv', r=0, c=0, label="E2", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B A2.csv', r=1, c=0, label="A2", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B D3.csv', r=2, c=0, label="D3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B G3.csv', r=0, c=1, label="G3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B B3.csv', r=1, c=1, label="B3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E4.csv', r=2, c=1, label="E4", coils=coils)

    for ax in axs:
        for killme in ax:
            killme.set_ylim(0, 60)

    fig.suptitle(f"coils={coils}")

    plt.tight_layout(pad=1)

    plt.savefig(fr"Output-Data\Images\{coils}-100")

for coils in range(50, 110, 10):
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E2.csv', r=0, c=0, label="E2", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B A2.csv', r=1, c=0, label="A2", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B D3.csv', r=2, c=0, label="D3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B G3.csv', r=0, c=1, label="G3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B B3.csv', r=1, c=1, label="B3", coils=coils)
    plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E4.csv', r=2, c=1, label="E4", coils=coils)

    for ax in axs:
        for killme in ax:
            killme.set_ylim(0, 60)

    fig.suptitle(f"coils={coils}")

    plt.tight_layout(pad=1)

    plt.savefig(fr"Output-Data\Images\{coils}-100")

# coils = 5

# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E2.csv', index=0, label="E2")
# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B A2.csv', index=1, label="A2")
# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B D3.csv', index=2, label="D3")
# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B G3.csv', index=3, label="G3")
# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B B3.csv', index=4, label="B3")
# plot_txt(fig, axs, fr'Audacity-Data\Hybrid-Humbucker\{coils}-100\Hybrid {coils}-100 B E4.csv', index=5, label="E4")

# for ax in axs:
#     ax.set_ylim(0, 60)

# plt.tight_layout(pad=10)

# plt.savefig(fr"Output-Data\Images\{coils}-100")

# plt.show()