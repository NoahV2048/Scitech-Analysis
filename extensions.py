import os

def change_extensions_to_csv(directory):
    # Loop through each file in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue

        # Split the filename into name and current extension
        base, _ = os.path.splitext(filename)

        # Create the new filename with .csv extension
        new_filename = base + ".csv"
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(file_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")

# Example usage
directory_path = r"Audacity-Data\Hybrid-Humbucker\Noise"
change_extensions_to_csv(directory_path)
