# clustering_analysis.py | K-Means Clustering Analysis
# ------------------------------------------------------------------------------
# Performs k-means clustering on z-scored state variables to identify regimes.
# Evaluates K=2-6 using inertia and silhouette score, then visualizes results.
#
# Uses config.yaml for parameters and includes caching for efficiency.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import yaml
from pathlib import Path
import warnings

# Apply a professional plot style
sns.set_style("whitegrid")
warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------------------------------------------------------#
# 0  Config helpers
# -----------------------------------------------------------------------------#
def load_config():
    """Read parameters from config.yaml."""
    with open('config.yaml', 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

cfg = load_config()

# -----------------------------------------------------------------------------#
# 1  Load and inspect df_zscored
# -----------------------------------------------------------------------------#
print("Loading state variables data...")

# Import state_variables to get df_zscored
# We'll suppress plots by temporarily redirecting plt.show
import sys
import importlib.util

# Suppress plots by monkey-patching plt.show temporarily
_original_show = plt.show
plt.show = lambda *args, **kwargs: None

try:
    # Load state_variables module dynamically
    # Try to find the file relative to current file or workspace root
    src_dir = Path(__file__).resolve().parent.parent
    state_vars_path = src_dir / "state_variables" / "state_variables.py"
    if not state_vars_path.exists():
        state_vars_path = Path.cwd() / "src" / "state_variables" / "state_variables.py"
    
    spec = importlib.util.spec_from_file_location("state_variables", state_vars_path)
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)
    
    df_zscored = sv.df_zscored.copy()
    common_start_date = sv.common_start_date
    print(f"Successfully loaded df_zscored from state_variables.py")
except (ImportError, AttributeError, FileNotFoundError) as e:
    # If import fails, try alternative import method
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from state_variables import state_variables as sv
        df_zscored = sv.df_zscored.copy()
        common_start_date = sv.common_start_date
        print(f"Successfully loaded df_zscored from state_variables.py (alternative method)")
    except Exception as e2:
        print(f"Error loading df_zscored: {e2}")
        print("Please ensure state_variables.py has been run at least once.")
        sys.exit(1)
finally:
    plt.show = _original_show

# Cut df_zscored to common start date (same as df_winsorized)
df_zscored = df_zscored.loc[common_start_date:].copy()

# Drop any remaining NaN rows
df_zscored = df_zscored.dropna()

print(f"\nDataFrame structure inspection:")
print(f"Shape: {df_zscored.shape} (rows: months, columns: state variables)")
print(f"Index type: {type(df_zscored.index)}")
print(f"Date range: {df_zscored.index.min()} to {df_zscored.index.max()}")
print(f"\nColumns (state variables):")
for i, col in enumerate(df_zscored.columns, 1):
    print(f"  {i}. {col}")
print(f"\nFirst few rows:")
print(df_zscored.head())
print(f"\nDataFrame info:")
df_zscored.info()
print(f"\nSummary statistics:")
print(df_zscored.describe())

# -----------------------------------------------------------------------------#
# 2  K-Means Clustering Analysis
# -----------------------------------------------------------------------------#
print("\n" + "="*70)
print("Performing K-Means clustering analysis for K = 2-6...")
print("="*70)

# Prepare data for clustering (remove index, keep only values)
X = df_zscored.values
dates = df_zscored.index

# Standardize the data (though it's already z-scored, this ensures consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different values of K
clustering_config = cfg.get('analysis', {}).get('clustering_analysis', {})
k_min = clustering_config.get('k_min', 2)
k_max = clustering_config.get('k_max', 6)
K_range = range(k_min, k_max + 1)
inertias = []
silhouette_scores = []
kmeans_models = {}

random_state = clustering_config.get('random_state', 42)
n_init = clustering_config.get('n_init', 10)
max_iter = clustering_config.get('max_iter', 300)

for k in K_range:
    print(f"\nTesting K = {k}...")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    
    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    kmeans_models[k] = {'model': kmeans, 'labels': labels}
    
    print(f"  Inertia: {inertia:.2f}")
    print(f"  Silhouette score: {silhouette:.4f}")

# Choose best K based on silhouette score (higher is better)
best_k_silhouette = K_range[np.argmax(silhouette_scores)]
print(f"\n{'='*70}")
print(f"Best K based on silhouette score: {best_k_silhouette} (score: {silhouette_scores[np.argmax(silhouette_scores)]:.4f})")

# Also identify elbow point for inertia (look for largest decrease)
inertia_diffs = np.diff(inertias)
elbow_k_idx = np.argmax(inertia_diffs) + 1  # +1 because diff reduces length by 1
best_k_elbow = list(K_range)[elbow_k_idx] if elbow_k_idx < len(K_range) else best_k_silhouette
print(f"Elbow point (largest inertia decrease): K = {best_k_elbow}")

# Use silhouette score as primary criterion
best_k = best_k_silhouette
print(f"\nSelected K = {best_k} for final clustering")

# Get final model and labels
final_model = kmeans_models[best_k]['model']
final_labels = kmeans_models[best_k]['labels']

# Add cluster labels to DataFrame
df_zscored_with_clusters = df_zscored.copy()
df_zscored_with_clusters['Cluster'] = final_labels

print(f"\nCluster distribution:")
cluster_counts = pd.Series(final_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    pct = 100 * count / len(final_labels)
    print(f"  Cluster {cluster_id}: {count} months ({pct:.1f}%)")

# -----------------------------------------------------------------------------#
# 3  Visualizations and Analysis
# -----------------------------------------------------------------------------#
print("\n" + "="*70)
print("Generating visualizations and analysis...")
print("="*70)

reports_dir = Path(cfg['paths']['reports_dir']) / 'analysis' / 'clustering analysis'
reports_dir.mkdir(parents=True, exist_ok=True)

# 3.1 Plot inertia and silhouette scores
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Inertia plot
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=best_k_elbow, color='r', linestyle='--', alpha=0.7, label=f'Elbow: K={best_k_elbow}')
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia', fontsize=12)
ax1.set_title('K-Means Inertia by Number of Clusters', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(K_range)

# Silhouette score plot
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best: K={best_k}')
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Silhouette Score by Number of Clusters', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(K_range)

plt.tight_layout()
plt.savefig(reports_dir / 'clustering_evaluation.png', dpi=300, bbox_inches='tight')
print("Saved: clustering_evaluation.png")

# 3.2 PCA 2D scatter plot colored by cluster
print("\nGenerating PCA 2D scatter plot...")

# Perform PCA
clustering_config = cfg.get('analysis', {}).get('clustering_analysis', {})
pca_n_components = clustering_config.get('pca_n_components', 2)
pca = PCA(n_components=pca_n_components, random_state=random_state)
X_pca = pca.fit_transform(X_scaled)

# Create scatter plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, 
                     cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster', fontsize=12)

plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.1%})', 
           fontsize=12)
plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.1%})', 
           fontsize=12)
plt.title(f'PCA 2D Scatter Plot: State Variables Colored by Cluster (K={best_k})', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Add cluster centers in PCA space
cluster_centers_pca = pca.transform(final_model.cluster_centers_)
plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
           c='red', marker='X', s=200, edgecolors='black', linewidth=2, 
           label='Cluster Centers', zorder=5)
plt.legend(fontsize=10)

plt.tight_layout()
plt.savefig(reports_dir / 'clustering_pca_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: clustering_pca_scatter.png")

# 3.3 Table of cluster mean z-scores
print("\nGenerating cluster mean z-scores table...")

cluster_means = df_zscored_with_clusters.groupby('Cluster').mean()
cluster_means = cluster_means.T  # Transpose so variables are rows, clusters are columns

# Create a styled table visualization
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=cluster_means.round(3).values,
                rowLabels=cluster_means.index,
                colLabels=[f'Cluster {i}' for i in cluster_means.columns],
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

# Style cells based on role/content
cell_dict = table.get_celld()
row_label_text = set(cluster_means.index)
col_label_text = {f'Cluster {i}' for i in cluster_means.columns}

for (row_idx, col_idx), cell in cell_dict.items():
    text = cell.get_text().get_text()
    
    # Header row (cluster labels)
    if row_idx == 0 and text in col_label_text:
        cell.set_facecolor('#E6E6E6')
        cell.set_text_props(weight='bold', size=12)
        continue
    
    # Top-left blank corner
    if row_idx == 0 and text == '':
        cell.set_facecolor('#E6E6E6')
        continue
    
    # Row labels (state variables)
    if text in row_label_text:
        cell.set_facecolor('#F0F0F0')
        cell.set_text_props(weight='bold', size=11, ha='left')
        continue
    
    # Numeric data cells
    try:
        value = float(text)
    except (TypeError, ValueError):
        continue
    
    if value < -0.5:
        cell.set_facecolor('#ffcccc')  # Light red
    elif value < 0:
        cell.set_facecolor('#ffe6e6')  # Very light red
    elif value == 0:
        cell.set_facecolor('#ffffff')  # White
    elif value < 0.5:
        cell.set_facecolor('#e6ffe6')  # Very light green
    else:
        cell.set_facecolor('#ccffcc')  # Light green
    
    cell.set_text_props(weight='bold')

plt.title(f'Cluster Mean Z-Scores by State Variable (K={best_k})', 
          fontsize=14, pad=20, fontweight='bold')
plt.savefig(reports_dir / 'clustering_means_table.png', dpi=300, bbox_inches='tight')
print("Saved: clustering_means_table.png")

# Also print the table to console
print("\n" + "="*70)
print("Cluster Mean Z-Scores Table:")
print("="*70)
print(cluster_means.round(3))

# -----------------------------------------------------------------------------#
# 4  Interpretation
# -----------------------------------------------------------------------------#
print("\n" + "="*70)
print("CLUSTERING INTERPRETATION")
print("="*70)

# Calculate average silhouette score
avg_silhouette = silhouette_scores[best_k - 2]  # -2 because K_range starts at 2

# Calculate within-cluster sum of squares relative to total
total_inertia = np.sum((X_scaled - X_scaled.mean(axis=0))**2)
within_cluster_ss = final_model.inertia_
between_cluster_ss = total_inertia - within_cluster_ss
explained_variance_ratio = between_cluster_ss / total_inertia

# Calculate separation between clusters (average distance between cluster centers)
cluster_centers = final_model.cluster_centers_
center_distances = cdist(cluster_centers, cluster_centers)
# Remove diagonal (distance to self)
np.fill_diagonal(center_distances, np.nan)
avg_center_distance = np.nanmean(center_distances)

# Calculate average within-cluster distance
avg_within_cluster_distances = []
for cluster_id in range(best_k):
    cluster_points = X_scaled[final_labels == cluster_id]
    if len(cluster_points) > 1:
        cluster_distances = cdist(cluster_points, [cluster_centers[cluster_id]])
        avg_within_cluster_distances.append(np.mean(cluster_distances))
avg_within_dist = np.mean(avg_within_cluster_distances) if avg_within_cluster_distances else 0

separation_ratio = avg_center_distance / avg_within_dist if avg_within_dist > 0 else 0

print(f"\nSelected K = {best_k} clusters")
print(f"Average silhouette score: {avg_silhouette:.4f}")
print(f"Explained variance ratio: {explained_variance_ratio:.2%}")
print(f"Average separation between cluster centers: {avg_center_distance:.3f}")
print(f"Average within-cluster distance: {avg_within_dist:.3f}")
print(f"Separation ratio (between/within): {separation_ratio:.2f}")

print("\nInterpretation:")
if avg_silhouette > 0.5:
    cluster_quality = "clear, well-separated clusters"
elif avg_silhouette > 0.3:
    cluster_quality = "moderately distinct clusters with some overlap"
elif avg_silhouette > 0.1:
    cluster_quality = "weak clustering with significant overlap"
else:
    cluster_quality = "very diffuse cloud with minimal clustering structure"

if separation_ratio > 2.0:
    separation_desc = "good separation"
elif separation_ratio > 1.5:
    separation_desc = "moderate separation"
else:
    separation_desc = "limited separation"

print(f"\nThe state observations form {cluster_quality}.")
print(f"The clustering shows {separation_desc} between clusters (ratio: {separation_ratio:.2f}).")

if avg_silhouette < 0.3:
    print("\nNote: The relatively low silhouette score suggests that the state variables")
    print("may form a more continuous distribution rather than distinct regimes.")
    print("This could indicate that economic states transition gradually rather than")
    print("through discrete regime shifts.")
elif avg_silhouette >= 0.5:
    print("\nThe high silhouette score indicates that the identified clusters represent")
    print("distinct economic regimes with meaningful differences in state variable values.")
else:
    print("\nThe moderate silhouette score suggests some clustering structure exists,")
    print("but there is overlap between regimes, indicating gradual transitions.")

print("\n" + "="*70)
print(f"All outputs saved to: {reports_dir}")
print("="*70)

