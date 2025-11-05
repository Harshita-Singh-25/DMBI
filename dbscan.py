import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

# ========================================
# CONFIGURATION
# ========================================
# SCENARIO 1: Mall Customer Dataset
DATASET_FILE = '/content/Mall_Customers.csv'
FEATURES_TO_DROP = ['CustomerID', 'Gender']
# --- Parameters to Tune ---
DBSCAN_EPS = 0.5       # Epsilon (radius)
DBSCAN_MIN_SAMPLES = 5   # Minimum points
# --- Visualization ---
VIZ_FEATURE_1 = 'Annual Income (k$)'
VIZ_FEATURE_2 = 'Spending Score (1-100)'

# --- 1. Load Data ---
print("="*50)
print("DBSCAN CLUSTERING")
print("="*50)
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded '{DATASET_FILE}'. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found.")
    exit()

# --- 2. Preprocess Data ---
# Select only features for clustering
try:
    X = df.drop(columns=FEATURES_TO_DROP, errors='ignore')
except KeyError:
    print("Error: Could not drop specified columns.")
    exit()

# One-Hot Encode categorical features (if any)
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
if X.isnull().sum().any():
    print("Handling missing values using 'mean' strategy...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# --- 3. Scale Data (Essential for DBSCAN) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ Data scaled successfully.\n")

# --- 4. Build DBSCAN Model ---
print(f"Building DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
dbscan.fit(X_scaled)

# Add cluster labels back to the original DataFrame
labels = dbscan.labels_
df['Cluster'] = labels

# --- 5. Model Evaluation ---
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("="*50)
print("CLUSTERING RESULTS")
print("="*50)
print(f"Number of Clusters Found: {n_clusters}")
print(f"Number of Noise Points (Outliers): {n_noise}")

# Calculate Silhouette Score (only on non-noise points)
if n_clusters > 1:
    non_noise_mask = (labels != -1)
    score = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
    print(f"Silhouette Score (for non-noise points): {score:.4f}")
else:
    print("Silhouette Score not calculated (need at least 2 clusters).")

# Show cluster distribution
print("\nCluster Distribution:")
cluster_counts = pd.Series(labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    cluster_name = "Noise (-1)" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"  {cluster_name}: {count} points")

# --- 6. Visualize Clusters ---
try:
    plt.figure(figsize=(12, 6))

    # Plot 1: Scatter plot of clusters
    plt.subplot(1, 2, 1)
    sns.scatterplot(
        x=VIZ_FEATURE_1,
        y=VIZ_FEATURE_2,
        hue='Cluster',
        data=df,
        palette='Spectral', # 'Spectral' is great for this
        style='Cluster',     # Use different markers
        s=100,
        alpha=0.7,
        edgecolor='k'
    )
    plt.title(f'DBSCAN Clustering\n(eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Cluster size distribution
    plt.subplot(1, 2, 2)
    cluster_names = ['Noise' if x == -1 else f'C{x}' for x in cluster_counts.index]
    plt.bar(cluster_names, cluster_counts.values,
            color='teal', edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Points')
    plt.title('Cluster Size Distribution')

    plt.tight_layout()
    plt.savefig('dbscan_clustering.png', dpi=300)
    plt.show()
    print("\n✅ Visualization saved as 'dbscan_clustering.png'")

except KeyError:
    print(f"\nError: Could not visualize. Make sure '{VIZ_FEATURE_1}' and '{VIZ_FEATURE_2}' are in the data.")
except Exception as e:
    print(f"\nAn error occurred during visualization: {e}")

# --- 7. Parameter Tuning Tips ---
print("\n" + "="*50)
print("PARAMETER TUNING TIPS")
print("="*50)
print("• If you get too much 'Noise': Increase 'eps' or Decrease 'min_samples'.")
print("• If you get one giant cluster: Decrease 'eps' or Increase 'min_samples'.")
print("\n✅ DBSCAN Clustering Completed Successfully!")