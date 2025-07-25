import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def weighted_calc(path):
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

    # Frequency denominations
    # print(np.mean(np.diff(freqs)))

    fft_offset = 50

    positive_freqs = freqs[fft_offset:N // 2]
    positive_magnitudes = np.abs(fft_vals[fft_offset:N // 2])
    return (np.sum(positive_freqs * positive_magnitudes) / np.sum(positive_magnitudes))

for idx in range(0, 50, 5):
    print(weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 B Strum.csv"))

for idx in range(50, 110, 10):
    print(weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 B Strum.csv"))

print()

for idx in range(0, 50, 5):
    print(weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 N Strum.csv"))

for idx in range(50, 110, 10):
    print(weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 N Strum.csv"))

print()

for idx in range(0, 50, 5):
    print(weighted_calc(F"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {idx}-100 Noise.csv"))

for idx in range(50, 110, 10):
    print(weighted_calc(F"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {idx}-100 Noise.csv"))

exit(0)

for idx in range(0, 50, 5):
    print(f"{idx}-100 B:", weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 B Strum.csv"))

for idx in range(50, 110, 10):
    print(f"{idx}-100 B:", weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 B Strum.csv"))

print()

for idx in range(0, 50, 5):
    print(f"{idx}-100 N:", weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 N Strum.csv"))

for idx in range(50, 110, 10):
    print(f"{idx}-100 N:", weighted_calc(F"Audacity-Data\Hybrid-Humbucker\{idx}-100\Hybrid {idx}-100 N Strum.csv"))

print()

for idx in range(0, 50, 5):
    print(f"{idx}-100 Noise:", weighted_calc(F"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {idx}-100 Noise.csv"))

for idx in range(50, 110, 10):
    print(f"{idx}-100 Noise:", weighted_calc(F"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {idx}-100 Noise.csv"))

print()

plt.show()