# ===============================================
# K-Means Clustering
# ===============================================
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# --- CONFIGURATION: Change these for a new dataset ---
# SCENARIO 1: Mall Customer Dataset
DATASET_FILE = 'Mall_Customers.csv'
FEATURES_TO_DROP = ['CustomerID', 'Gender'] # Drop ID and categorical for this example
N_CLUSTERS = 5
VISUALIZATION_FEATURE_1 = 'Annual Income (k$)'
VISUALIZATION_FEATURE_2 = 'Spending Score (1-100)'

# # SCENARIO 2: Iris Dataset
# DATASET_FILE = 'Iris.csv'
# FEATURES_TO_DROP = ['Id', 'Species']
# N_CLUSTERS = 3
# VISUALIZATION_FEATURE_1 = 'PetalLengthCm'
# VISUALIZATION_FEATURE_2 = 'PetalWidthCm'
# --- END CONFIGURATION ---

# --- 1. Preprocess data (Scaling is essential) ---
print("Loading data...")
try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print(f"Error: '{DATASET_FILE}' not found.")
    exit()

X = df.drop(columns=[col for col in FEATURES_TO_DROP if col in df.columns], errors='ignore')
X = pd.get_dummies(X, drop_first=True)

if X.isnull().sum().any():
    print(f"\nHandling {X.isnull().sum().sum()} missing values (mean strategy)...")
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_imputed = imputer.fit_transform(X.values)
    X = pd.DataFrame(X_imputed, columns=X.columns)
else:
    print("No missing values found.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Data scaled using StandardScaler.")

# --- 2. Build Clustering model ---
print(f"Building K-Means model with n_clusters={N_CLUSTERS}...")
kmeans = KMeans(n_clusters=N_CLUSTERS, init='k-means++', random_state=42, n_init='auto')
kmeans.fit(X_scaled)

labels = kmeans.labels_
df['Cluster'] = labels # Add labels back for plotting

# --- 3. Determine Performance parameters ---
inertia = kmeans.inertia_
score = silhouette_score(X_scaled, labels)

print("\n--- Model Evaluation ---")
print(f"K-Means Inertia (WCSS): {inertia:.4f}")
print(f"Silhouette Score: {score:.4f}")

# --- 4. Visualize the Clusters ---
try:
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=VISUALIZATION_FEATURE_1,
        y=VISUALIZATION_FEATURE_2,
        hue='Cluster',
        data=df,
        palette='viridis',
        s=100
    )
    plt.title(f'K-Means Clustering: {N_CLUSTERS} Clusters')
    plt.xlabel(VISUALIZATION_FEATURE_1)
    plt.ylabel(VISUALIZATION_FEATURE_2)

    # Plotting cluster centers
    # Note: We need to find the column indices in the scaled data for the viz features
    idx1 = X.columns.get_loc(VISUALIZATION_FEATURE_1)
    idx2 = X.columns.get_loc(VISUALIZATION_FEATURE_2)

    # Get centers from the scaled space
    centers_scaled = kmeans.cluster_centers_

    # Plot centers
    plt.scatter(
        centers_scaled[:, idx1],
        centers_scaled[:, idx2],
        marker='X', s=250, color='red', label='Centroids', edgecolors='black'
    )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("kmeans_cluster_visualization.png")
    print("\n K-Means cluster visualization saved as 'kmeans_cluster_visualization.png'")
except Exception as e:
    print(f"\nError during visualization: {e}")