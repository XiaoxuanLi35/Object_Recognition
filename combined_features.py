from sklearn.cluster import KMeans
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from sklearn.decomposition import PCA
from collections import Counter
import os

# Cluster 2 had only 1 item, which is unbalanced

# Set environment variable to avoid CPU count warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

def reassign_small_clusters(clusters, min_samples=5, features=None):
    """
    Reassign samples to ensure each cluster has at least the minimum number of samples
    while maintaining balance between clusters.
    
    Parameters:
    -----------
    clusters : array-like
        Original cluster labels
    min_samples : int
        Minimum number of samples per cluster
    features : array-like
        Feature matrix for calculating distances to cluster centers
    """
    # Calculate sample count for each cluster
    cluster_counts = Counter(clusters)
    n_clusters = len(cluster_counts)
    
    # Create new cluster labels array
    new_clusters = clusters.copy()
    
    # Calculate cluster centers if features are provided
    if features is not None:
        centroids = np.array([features[clusters == i].mean(axis=0) for i in range(n_clusters)])
    
    # Find clusters that need reassignment (fewer than min_samples)
    small_clusters = [c for c, count in cluster_counts.items() if count < min_samples]
    
    if small_clusters:
        # Calculate target samples per cluster (average distribution)
        total_samples = len(clusters)
        target_samples = total_samples // n_clusters
        
        # Process each small cluster
        for small_cluster in small_clusters:
            # Calculate how many samples are needed
            needed_samples = min_samples - cluster_counts[small_cluster]
            
            # Find clusters with more than target samples
            large_clusters = [c for c, count in cluster_counts.items() 
                            if count > target_samples and c != small_cluster]
            
            if not large_clusters:
                continue
                
            # Reassign samples from large clusters
            for large_cluster in large_clusters:
                if needed_samples <= 0:
                    break
                    
                # Get samples from large cluster
                large_cluster_samples = np.where(new_clusters == large_cluster)[0]
                
                if features is not None:
                    # Calculate distances to small cluster center
                    distances = np.linalg.norm(features[large_cluster_samples] - 
                                            centroids[small_cluster], axis=1)
                    # Select closest samples
                    closest_samples = large_cluster_samples[np.argsort(distances)][:needed_samples]
                else:
                    # Random selection if no features provided
                    closest_samples = np.random.choice(large_cluster_samples, 
                                                     min(needed_samples, len(large_cluster_samples)), 
                                                     replace=False)
                
                # Reassign samples
                new_clusters[closest_samples] = small_cluster
                needed_samples -= len(closest_samples)
                
                # Update counts
                cluster_counts[large_cluster] -= len(closest_samples)
                cluster_counts[small_cluster] += len(closest_samples)
    
    return new_clusters

# Set paths
color_hist_path = Path(r"C:\Users\李晓璇\Desktop\Semester 3\CS6123\project_3\scikit-image\Cooking\color_histogram")
hog_path = Path(r"C:\Users\李晓璇\Desktop\Semester 3\CS6123\project_3\scikit-image\Cooking\HOG")

# Load color histogram features
color_histograms = []
color_filenames = []
for hist_path in color_hist_path.glob('*_histogram.npy'):
    hist = np.load(str(hist_path))
    color_histograms.append(hist)
    color_filenames.append(hist_path.stem.replace('_histogram', ''))

# Load HOG features
hog_features = []
hog_filenames = []
for hog_path in hog_path.glob('hog_*.png'):
    hog_img = imread(str(hog_path))
    if len(hog_img.shape) > 2:
        if hog_img.shape[2] == 4:
            hog_img = rgba2rgb(hog_img)
        hog_img = rgb2gray(hog_img)
    hog_feature = hog_img.flatten()
    hog_features.append(hog_feature)
    hog_filenames.append(hog_path.stem.replace('hog_', ''))

# Convert to numpy arrays
color_histograms = np.array(color_histograms)
hog_features = np.array(hog_features)

print("Original feature dimensions:")
print(f"Color histogram features shape: {color_histograms.shape}")
print(f"HOG features shape: {hog_features.shape}")

# Standardize features
scaler_color = StandardScaler()
scaler_hog = StandardScaler()
color_histograms_scaled = scaler_color.fit_transform(color_histograms)
hog_features_scaled = scaler_hog.fit_transform(hog_features)

# PCA for color histogram features
pca_color = PCA(n_components=0.90)  # Retain 90% variance
color_pca = pca_color.fit_transform(color_histograms_scaled)
print(f"\nColor histogram dimensions after PCA: {color_pca.shape}")
print(f"Variance ratio retained (color): {sum(pca_color.explained_variance_ratio_):.4f}")
print(f"Explained variance ratios (color): {pca_color.explained_variance_ratio_}")

# PCA for HOG features
pca_hog = PCA(n_components=0.90)
hog_pca = pca_hog.fit_transform(hog_features_scaled)
print(f"\nHOG features dimensions after PCA: {hog_pca.shape}")
print(f"Variance ratio retained (HOG): {sum(pca_hog.explained_variance_ratio_):.4f}")
print(f"Number of components needed (HOG): {hog_pca.shape[1]}")

# Combine features with weights
color_weight = 0.6  # Color feature weight
hog_weight = 0.4    # HOG feature weight
combined_features = np.concatenate([
    color_weight * color_pca,
    hog_weight * hog_pca
], axis=1)

print(f"\nCombined feature dimensions: {combined_features.shape}")

# K-means clustering
n_clusters = 3               # Number of clusters
min_samples_per_cluster = 3  # Minimum samples per cluster

# Initial clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
initial_clusters = kmeans.fit_predict(combined_features)

# Reassign small clusters
final_clusters = reassign_small_clusters(initial_clusters, 
                                       min_samples=min_samples_per_cluster,
                                       features=combined_features)

# Organize samples by cluster
cluster_samples = {i: [] for i in range(n_clusters)}
for filename, cluster_id in zip(color_filenames, final_clusters):
    cluster_samples[cluster_id].append(filename)

# Print results
print(f"\nClustering Results (minimum samples: {min_samples_per_cluster}):")
for cluster_id in range(n_clusters):
    samples = cluster_samples[cluster_id]
    print(f"\nCluster {cluster_id} ({len(samples)} samples):")
    for sample in samples:
        print(f"  - {sample}")

# Save results to file
output_file = "clustering_results_min_samples.txt"
with open(output_file, 'w') as f:
    f.write(f"Clustering Results (minimum samples: {min_samples_per_cluster}):\n")
    f.write(f"\nOriginal dimensions:\n")
    f.write(f"Color histogram: {color_histograms.shape}\n")
    f.write(f"HOG features: {hog_features.shape}\n")
    f.write(f"\nDimensions after PCA:\n")
    f.write(f"Color histogram: {color_pca.shape}\n")
    f.write(f"HOG features: {hog_pca.shape}\n")
    f.write(f"Combined features: {combined_features.shape}\n")
    
    f.write("\nClustering results:\n")
    for cluster_id in range(n_clusters):
        samples = cluster_samples[cluster_id]
        f.write(f"\nCluster {cluster_id} ({len(samples)} samples):\n")
        for sample in samples:
            f.write(f"  - {sample}\n")

print(f"\nResults saved to {output_file}")