import pandas as pd
import sqlite3
import numpy as np

# Define the database name we will use for both parts
DB_FILE = 'sales_dw.db'

print(f"Database file will be created at: {DB_FILE}\n")

# =======================================================
# PART 1: ETL (DESIGN & LOAD DATA WAREHOUSE)
# This part creates the 'sales_dw.db' file.
# =======================================================

print(f"--- PART 1: ETL PROCESS [STARTING] ---")

# 1. EXTRACT - Create Source Data (Mock DataFrames)
print("  (E) Extracting source data...")
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

# Mock transactional sales data
sales = pd.DataFrame({
    'sale_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    'date': pd.to_datetime(['2025-01-15', '2025-01-20', '2025-02-10',
                           '2025-02-15', '2025-03-05', '2025-03-10',
                           '2025-04-15', '2025-05-20']),
    'product_id': [1, 2, 3, 1, 4, 5, 1, 3],
    'store_id': [101, 101, 102, 103, 102, 103, 101, 102],
    'customer_id': [1, 2, 3, 4, 1, 2, 3, 4],
    'quantity': [2, 5, 3, 1, 2, 4, 1, 2]
})

# 2. TRANSFORM - Data Cleaning & Schema Creation
print("  (T) Transforming data into Star Schema...")

# Create 'total_amount' in the sales data
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

# Map date_id back to the main sales table
sales = sales.merge(dim_date[['date', 'date_id']], on='date')

# Create Final Dimension Tables (dim_)
dim_product = products
dim_store = stores
dim_customer = customers
# (dim_date already created)

# Create Final Fact Table (fact_)
fact_sales = sales[['sale_id', 'date_id', 'product_id',
                    'store_id', 'customer_id', 'quantity', 'total_amount']]

print(f"    - Fact Table created. Shape: {fact_sales.shape}")
print(f"    - Dimension Tables created: dim_product, dim_store, dim_customer, dim_date")

# 3. LOAD - Store in Data Warehouse (SQLite)
print(f"  (L) Loading tables into database: {DB_FILE}...")
conn = sqlite3.connect(DB_FILE)

# Load all tables into the SQLite database
dim_product.to_sql('dim_product', conn, if_exists='replace', index=False)
dim_store.to_sql('dim_store', conn, if_exists='replace', index=False)
dim_customer.to_sql('dim_customer', conn, if_exists='replace', index=False)
dim_date.to_sql('dim_date', conn, if_exists='replace', index=False)
fact_sales.to_sql('fact_sales', conn, if_exists='replace', index=False)

conn.close()


# =======================================================
# PART 2: OLAP (PERFORM OPERATIONS ON DATA WAREHOUSE)
# This part connects to the 'sales_dw.db' file that was just created.

print(f"\n--- PART 2: OLAP OPERATIONS [STARTING] ---")

# Connect to the newly created database
conn = sqlite3.connect(DB_FILE)

# Helper function to print queries nicely
def show_query(title, query):
    print(f"\n{'='*50}")
    print(f"OPERATION: {title}")
    print('='*50)
    try:
        df = pd.read_sql_query(query, conn)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"--- ERROR IN QUERY: {e} ---")

# --- 1. ROLL-UP (Summarize by Region) ---
query_rollup = """
SELECT region, SUM(total_amount) AS Total_Revenue
FROM fact_sales
JOIN dim_store USING (store_id)
GROUP BY region
ORDER BY Total_Revenue DESC;
"""
show_query("ROLL-UP: Total Revenue by Region", query_rollup)

# --- 2. DRILL-DOWN (Region → City) ---
query_drilldown = """
SELECT region, city, SUM(total_amount) AS City_Revenue
FROM fact_sales
JOIN dim_store USING (store_id)
GROUP BY region, city
ORDER BY region, City_Revenue DESC;
"""
show_query("DRILL-DOWN: Revenue by City within Region", query_drilldown)

# --- 3. SLICE (Filter by Quarter Q1) ---
query_slice = """
SELECT product_name, SUM(total_amount) AS Q1_Revenue
FROM fact_sales
JOIN dim_product USING (product_id)
JOIN dim_date USING (date_id)
WHERE quarter = 'Q1'
GROUP BY product_name
ORDER BY Q1_Revenue DESC;
"""
show_query("SLICE: Q1 Sales by Product", query_slice)

# --- 4. DICE (Electronics in North Region) ---
query_dice = """
SELECT product_name, city, SUM(total_amount) AS Revenue
FROM fact_sales
JOIN dim_product USING (product_id)
JOIN dim_store USING (store_id)
WHERE category = 'Electronics' AND region = 'North'
GROUP BY product_name, city
ORDER BY Revenue DESC;
"""
show_query("DICE: Electronics Sales in North Region", query_dice)

# --- 5. PIVOT (Product vs Quarter) ---
query_pivot = """
SELECT
  product_name,
  SUM(CASE WHEN quarter = 'Q1' THEN total_amount ELSE 0 END) AS Q1,
  SUM(CASE WHEN quarter = 'Q2' THEN total_amount ELSE 0 END) AS Q2
  -- Add Q3/Q4 if your data has them
FROM fact_sales
JOIN dim_product USING (product_id)
JOIN dim_date USING (date_id)
GROUP BY product_name;
"""
show_query("PIVOT: Product Revenue by Quarter", query_pivot)


# Clean up
conn.close()
print(f"\n--- ✅ PART 2: OLAP OPERATIONS [COMPLETED] ---")
print(f"\n--- 🏁 SCRIPT FINISHED ---")