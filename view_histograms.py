import numpy as np
from pathlib import Path

# Specify color_histogram folder path
histogram_path = Path(r"C:\Users\李晓璇\Desktop\Semester 3\CS6123\project_3\scikit-image\Forage\color_histogram")

# Read and view all histograms
for npy_file in histogram_path.glob('*.npy'):
   hist = np.load(npy_file)
   print(f"\nFilename: {npy_file.name}")
   print(f"Histogram shape: {hist.shape}")
   print(f"Histogram sum: {hist.sum():.2f}")