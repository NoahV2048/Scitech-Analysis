import os
import pandas as pd
import numpy as np

def prepend_correct_time_column(directory, sample_rate=44100):
    """
    Prepends a uniformly sampled time column to each file in the directory.
    Time is sampled at `sample_rate` for `duration_sec` seconds and added as the first column.
    """

    for filename in os.listdir(directory[:7]):
        if filename.endswith(".csv") or filename.endswith(".txt"):
            path = os.path.join(directory, filename)

            try:
                df = pd.read_csv(path, delimiter="\t", header=None)

                time_step = 1.0 / sample_rate
                num_samples = df.shape[0]

                # Truncate or pad to match the correct number of samples
                df = df.iloc[:num_samples].copy()

                # Create uniformly spaced time column
                time = np.round(np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False), 5)

                # Insert as the first column
                df.insert(0, "Time", time)

                # Save back to the file
                df.to_csv(path, sep="\t", index=False, header=False)
                print(f"Prepended time to: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

prepend_correct_time_column(r"Audacity-Data\Hybrid-Humbucker\Noise", sample_rate=44100, duration_sec=2.0)
