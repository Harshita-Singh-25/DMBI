import pandas as pd
import sqlite3
import numpy as np

# ========================================
# EXTRACT - Create Source Data
# ========================================

products = pd.DataFrame({
    'product_id': [1, 2, 3, 4, 5],
    'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headset'],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
    'price': [50000, 500, 1500, 15000, 2000]
})

stores = pd.DataFrame({
    'store_id': [101, 102, 103],
    'store_name': ['Mumbai Store', 'Pune Store', 'Delhi Store'],
    'city': ['Mumbai', 'Pune', 'Delhi'],
    'region': ['West', 'West', 'North']
})

customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'customer_name': ['Rahul', 'Priya', 'Amit', 'Sneha'],
    'age': [25, 30, 35, 28],
    'gender': ['M', 'F', 'M', 'F']
})

sales = pd.DataFrame({
    'sale_id': [1001, 1002, 1003, 1004, 1005, 1006],
    'date': pd.to_datetime(['2025-01-15', '2025-01-20', '2025-02-10',
                            '2025-02-15', '2025-03-05', '2025-03-10']),
    'product_id': [1, 2, 3, 1, 4, 5],
    'store_id': [101, 101, 102, 103, 102, 103],
    'customer_id': [1, 2, 3, 4, 1, 2],
    'quantity': [2, 5, 3, 1, 2, 4]
})

# ========================================
# TRANSFORM - Data Cleaning & Enrichment
# ========================================

# Add total_amount column
sales = sales.merge(products[['product_id', 'price']], on='product_id')
sales['total_amount'] = sales['quantity'] * sales['price']

# Create Date Dimension
dim_date = pd.DataFrame({
    'date': sales['date'].unique()
})
dim_date['date_id'] = range(1, len(dim_date) + 1)
dim_date['day'] = dim_date['date'].dt.day
dim_date['month'] = dim_date['date'].dt.month
dim_date['year'] = dim_date['date'].dt.year
dim_date['quarter'] = 'Q' + dim_date['date'].dt.quarter.astype(str)

# Map date_id to fact table
sales = sales.merge(dim_date[['date', 'date_id']], on='date')

# Create Fact Table
fact_sales = sales[['sale_id', 'date_id', 'product_id',
                     'store_id', 'customer_id', 'quantity', 'total_amount']]

# ========================================
# LOAD - Store in Data Warehouse
# ========================================

conn = sqlite3.connect('sales_dw.db')

products.to_sql('dim_product', conn, if_exists='replace', index=False)
stores.to_sql('dim_store', conn, if_exists='replace', index=False)
customers.to_sql('dim_customer', conn, if_exists='replace', index=False)
dim_date.to_sql('dim_date', conn, if_exists='replace', index=False)
fact_sales.to_sql('fact_sales', conn, if_exists='replace', index=False)

print("✅ ETL Process Completed Successfully!")
print(f"\nFact Table Shape: {fact_sales.shape}")
print(f"Total Revenue: ₹{fact_sales['total_amount'].sum():,.2f}")

# Display sample data
print("\n=== FACT TABLE SAMPLE ===")
print(fact_sales.head())

conn.close()