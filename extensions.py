import os

def change_extensions_to_csv(directory):
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        
        if os.path.isdir(file_path):
            continue

        
        base, _ = os.path.splitext(filename)

        
        new_filename = base + ".csv"
        new_path = os.path.join(directory, new_filename)

        
        os.rename(file_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")


directory_path = r"Audacity-Data\Hybrid-Humbucker\Noise"
change_extensions_to_csv(directory_path)
