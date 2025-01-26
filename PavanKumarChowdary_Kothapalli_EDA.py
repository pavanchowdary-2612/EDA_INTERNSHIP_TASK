
"""
Created on Mon Jan 27 01:21:44 2025

@author: pavan
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers_path = r"C:\Users\pavan\Downloads\Customers.csv"
products_path = r"C:\Users\pavan\Downloads\Products.csv"
transactions_path = r"C:\Users\pavan\Downloads\Transactions.csv"

customers_df = pd.read_csv(customers_path)
products_df = pd.read_csv(products_path)
transactions_df = pd.read_csv(transactions_path)

# Merge transactions with customers and products for comprehensive analysis
transactions_by_region = transactions_df.merge(customers_df, on="CustomerID")
transactions_with_products = transactions_df.merge(products_df, on="ProductID")

# Analyze regional activity
region_distribution = transactions_by_region['Region'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=region_distribution.index, y=region_distribution.values, palette="viridis")
plt.title("Transactions Distribution by Region")
plt.ylabel("Number of Transactions")
plt.xlabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze customer spending patterns
customer_spending = transactions_df.groupby("CustomerID")["TotalValue"].sum().sort_values(ascending=False)
top_customers = customer_spending.head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_customers.index, y=top_customers.values, palette="plasma")
plt.title("Top Customers by Total Spending")
plt.ylabel("Total Spending (USD)")
plt.xlabel("CustomerID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Analyze product popularity
product_popularity = transactions_df.groupby("ProductID")["Quantity"].sum().sort_values(ascending=False)
top_products = product_popularity.head(10).reset_index()
top_products = top_products.merge(products_df, on="ProductID")
plt.figure(figsize=(8, 5))
sns.barplot(x=top_products["ProductName"], y=top_products["Quantity"], palette="cividis")
plt.title("Top Products by Quantity Sold")
plt.ylabel("Quantity Sold")
plt.xlabel("Product Name")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Display basic statistics and check for missing values
customers_summary = customers_df.describe(include='all')
products_summary = products_df.describe(include='all')
transactions_summary = transactions_df.describe(include='all')
missing_values = {
    "Customers": customers_df.isnull().sum(),
    "Products": products_df.isnull().sum(),
    "Transactions": transactions_df.isnull().sum()
}

print("Summary of Customers Dataset:")
print(customers_summary)
print("\nSummary of Products Dataset:")
print(products_summary)
print("\nSummary of Transactions Dataset:")
print(transactions_summary)
print("\nMissing Values in Each Dataset:")
print(missing_values)