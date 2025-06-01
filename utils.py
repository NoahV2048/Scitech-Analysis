import os
import pandas as pd
import numpy as np

def prepend_correct_time_column(directory, sample_rate=44100):

    for filename in os.listdir(directory):
        # if filename != 'Hybrid 10-100 B B3.csv':
        #     continue

        if filename.endswith(".csv") or filename.endswith(".txt"):
            path = os.path.join(directory, filename)

            try:
                df = pd.read_csv(path, delimiter="\t", header=None)

                time_step = 1.0 / sample_rate
                num_samples = df.shape[0]

                
                df = df.iloc[:num_samples].copy()

                
                time = np.round(np.linspace(0, num_samples / sample_rate, num_samples, endpoint=False), 5)

                
                df.insert(0, "Time", time)

                
                df.to_csv(path, sep="\t", index=False, header=False)
                print(f"Prepended time to: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

prepend_correct_time_column(r"Audacity-Data\Hybrid-Humbucker\10-100", sample_rate=44100)
