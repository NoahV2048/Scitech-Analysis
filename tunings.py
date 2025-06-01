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

def plot_txt(path: str):
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

    return harmonic_peaks_sorted[0][0]

for i in ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']:
    std_ev = np.array([])
    for j in range(0, 50, 5):
        res = plot_txt(f"Audacity-Data\Hybrid-Humbucker\{j}-100\Hybrid {j}-100 N {i}.csv")
        if res == 48.99993366794482:
            continue
        # print(res)
        std_ev = np.append(std_ev, res)
        
    for j in range(50, 110, 10):
        res = plot_txt(f"Audacity-Data\Hybrid-Humbucker\{j}-100\Hybrid {j}-100 N {i}.csv")
        # print(res)
        std_ev = np.append(std_ev, res)

    print(np.std(std_ev), np.mean(std_ev))