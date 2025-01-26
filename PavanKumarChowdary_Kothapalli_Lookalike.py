# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 01:35:07 2025

@author: pavan
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

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
# Aggregate total spending, quantity, and category purchase counts per customer
customer_profiles = merged_data.groupby("CustomerID").agg(
    total_spent=("TotalValue", "sum"),
    total_quantity=("Quantity", "sum"),
    **{f"category_{cat}": ("Category", lambda x: (x == cat).sum()) for cat in products_df["Category"].unique()}
).reset_index()

# Normalize the features for similarity calculations
scaler = StandardScaler()
profile_features = customer_profiles.drop(columns=["CustomerID"])
normalized_features = scaler.fit_transform(profile_features)

# Compute cosine similarity
similarity_matrix = cosine_similarity(normalized_features)

# Get the top 3 most similar customers for the first 20 customers
top_lookalikes = {}
for i, customer_id in enumerate(customer_profiles["CustomerID"][:20]):
    similarities = list(enumerate(similarity_matrix[i]))
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_matches = [(customer_profiles.loc[idx, "CustomerID"], round(score, 3))
                   for idx, score in sorted_similarities[1:4]]
    top_lookalikes[customer_id] = top_matches

# Create Lookalike CSV file
lookalike_df = pd.DataFrame({
    "cust_id": top_lookalikes.keys(),
    "lookalikes": [str(matches) for matches in top_lookalikes.values()]
})
lookalike_csv_path = r"C:\Users\pavan\OneDrive\Lookalike.csv"
lookalike_df.to_csv(lookalike_csv_path, index=False)

print(f"Lookalike CSV created at: {lookalike_csv_path}")
