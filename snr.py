import pandas as pd
import numpy as np

def compute_snr(signal, noise):
    """
    Computes SNR in decibels from two segments: signal and noise (in linear amplitude).

    Args:
        signal (np.ndarray): Signal segment (linear scale).
        noise (np.ndarray): Noise segment (linear scale).

    Returns:
        float: SNR in dB
    """
    signal_power = np.sqrt(np.mean(signal ** 2))
    print("Average Signal Power", signal_power)

    noise_power = np.sqrt(np.mean(noise ** 2))
    print("Average Noise Power", noise_power)

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def compute_snr_from_files(signalfile, noisefile):
    signal_df = pd.read_csv(signalfile, delimiter="\t", header=None)
    filtered_signal_df = np.array(signal_df[signal_df[1] != '[-inf]'], dtype=float)
    noise_df = pd.read_csv(noisefile, delimiter="\t", header=None)
    filtered_noise_df = np.array(noise_df[noise_df[1] != '[-inf]'], dtype=float)

    return compute_snr(10 ** (filtered_signal_df[1] / 20.0), 10 ** (filtered_noise_df[1] / 20.0))

def get_snr_linear(snr_db):
    return snr_db
    # return 10 ** (snr_db / 20)

print(get_snr_linear(compute_snr_from_files(
    f"Audacity-Data\\Hybrid-Humbucker\\30-100\\Hybrid 30-100 N Strum.csv", 
    f"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid 30-100 Noise.csv"
)))

exit(0)

for i in range(0, 50, 5):
    print(get_snr_linear(compute_snr_from_files(
        f"Audacity-Data\\Hybrid-Humbucker\\{i}-100\\Hybrid {i}-100 N Strum.csv", 
        f"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {i}-100 Noise.csv"
    )))

for i in range(50, 100, 10):
    print(get_snr_linear(compute_snr_from_files(
        f"Audacity-Data\\Hybrid-Humbucker\\{i}-100\\Hybrid {i}-100 B Strum.csv", 
        f"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {i}-100 Noise.csv"
    )))

print()

for i in range(0, 50, 5):
    print(compute_snr_from_files(
        f"Audacity-Data\\Hybrid-Humbucker\\{i}-100\\Hybrid {i}-100 N Strum.csv", 
        f"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {i}-100 Noise.csv"
    ))

for i in range(50, 100, 10):
    print(compute_snr_from_files(
        f"Audacity-Data\\Hybrid-Humbucker\\{i}-100\\Hybrid {i}-100 N Strum.csv", 
        f"Audacity-Data\\Hybrid-Humbucker\\Noise\\Hybrid {i}-100 Noise.csv"
    ))

