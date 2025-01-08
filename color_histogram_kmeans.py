from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path

# Set path to histogram data directory
output_path = Path(r"C:\Users\name\Desktop\Semester 3\CS6123\project_3\scikit-image\Cooking\color_histogram")

# Load all histogram data
histograms = []
for hist_path in output_path.glob('*_histogram.npy'):
   hist = np.load(str(hist_path))# reads the numerical data from the .npy file
   histograms.append(hist) # adds the loaded histogram to the list

# Convert to numpy array
X = np.array(histograms)

# K-means clustering (assuming 4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Get cluster assignments for each image
for hist_path, cluster_id in zip(output_path.glob('*_histogram.npy'), clusters):
   print(f"{hist_path.stem}: Cluster {cluster_id}")
