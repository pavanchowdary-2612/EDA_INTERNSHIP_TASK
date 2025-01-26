# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 02:02:56 2025

@author: pavan
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
customers_path = r"C:\Users\pavan\Downloads\Customers.csv"
products_path = r"C:\Users\pavan\Downloads\Products.csv"
transactions_path = r"C:\Users\pavan\Downloads\Transactions.csv"

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)

# Merge datasets for unified analysis
merged_data = transactions_df.merge(products_df, on="ProductID").merge(customers_df, on="CustomerID")

# Feature engineering: Customer profiles
customer_profiles = merged_data.groupby("CustomerID").agg(
    total_spent=("TotalValue", "sum"),
    total_quantity=("Quantity", "sum"),
    **{f"category_{cat}": ("Category", lambda x: (x == cat).sum()) for cat in products_df["Category"].unique()}
).reset_index()

# Normalize the features for clustering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
clustering_features = scaler.fit_transform(customer_profiles.drop(columns=["CustomerID"]))

# Determine the optimal number of clusters using Davies-Bouldin Index (2 to 10 clusters)
db_scores = []
range_n_clusters = range(2, 11)
for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_features)
    db_index = davies_bouldin_score(clustering_features, cluster_labels)
    db_scores.append(db_index)

# Optimal number of clusters
optimal_clusters = range_n_clusters[db_scores.index(min(db_scores))]

# Fit KMeans with the optimal number of clusters
final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(clustering_features)

# Add cluster labels to the customer profiles
customer_profiles["Cluster"] = final_labels

# Reduce dimensionality for visualization using PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(clustering_features)

# Plot clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    cluster_points = reduced_features[final_labels == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
plt.title("Customer Segments (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.tight_layout()
plt.show()

# Clustering summary
print("Optimal Number of Clusters:", optimal_clusters)
print("Davies-Bouldin Index (DB Index):", min(db_scores))
print("Cluster Distribution:")
print(customer_profiles.groupby("Cluster").size())