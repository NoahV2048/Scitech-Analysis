import pandas as pd
import numpy as np

def extract_multiples(input_file, output_file, num_multiples=10):
    
    df = pd.read_csv(input_file, delimiter="\t", header=None, names=["Freq", "Amplitude"])
    
    
    f0 = df["Freq"].iloc[0]

    
    results = []
    for n in range(1, num_multiples + 1):
        target_freq = n * f0
        idx_closest = (df["Freq"] - target_freq).abs().idxmin()
        closest_freq = df.loc[idx_closest, "Freq"]
        amplitude = df.loc[idx_closest, "Amplitude"]
        results.append((closest_freq, amplitude))

    
    output_df = pd.DataFrame(results, columns=["Closest_Freq", "Amplitude"])
    output_df.to_csv(output_file, sep="\t", index=False)

for i in range(0, 50, 5):
    path = f'Hybrid {i}-100 B A2.csv'
    extract_multiples(f'Output-Data\{path}', f'Multiples_Data\{path}')

for i in range(50, 100, 10):
    path = f'Hybrid {i}-100 B A2.csv'
    extract_multiples(f'Output-Data\{path}', f'Multiples_Data\{path}')
