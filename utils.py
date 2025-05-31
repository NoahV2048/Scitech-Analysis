import os
import pandas as pd
import numpy as np

def prepend_correct_time_column(directory, sample_rate=44100, duration_sec=2.0):
    """
    Prepends a uniformly sampled time column to each file in the directory.
    Time is sampled at `sample_rate` for `duration_sec` seconds and added as the first column.
    """
    time_step = 1.0 / sample_rate
    num_samples = int(sample_rate * duration_sec)

    for filename in os.listdir(directory):
        if filename.endswith(".csv") or filename.endswith(".txt"):
            path = os.path.join(directory, filename)

            try:
                df = pd.read_csv(path, delimiter="\t", header=None)

                if df.shape[0] < num_samples:
                    print(f"Skipping {filename} (only {df.shape[0]} rows, expected {num_samples})")
                    continue

                # Truncate or pad to match the correct number of samples
                df = df.iloc[:num_samples].copy()

                # Create uniformly spaced time column
                time = np.round(np.linspace(0, duration_sec, num_samples, endpoint=False), 5)

                # Insert as the first column
                df.insert(0, "Time", time)

                # Save back to the file
                df.to_csv(path, sep="\t", index=False, header=False)
                print(f"Prepended time to: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

prepend_correct_time_column(r"Audacity-Data\Hybrid-Humbucker\30-100", sample_rate=44100, duration_sec=2.0)
