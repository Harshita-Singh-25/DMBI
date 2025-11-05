import pandas as pd
import sqlite3

# ========================================
# 1. CONNECT & INSPECT
# ========================================

# Connect to the database file you uploaded
conn = sqlite3.connect('/content/sales.db')

print("Inspecting the database...")
table_list_query = "SELECT name FROM sqlite_master WHERE type='table';"
tables = pd.read_sql_query(table_list_query, conn)['name']

for table in tables:
    if table.startswith('sqlite_'):
        continue  # Skip internal sqlite tables

    print(f"\n{'='*30}")
    print(f"Table: {table}")
    print(f"{'='*30}")

    # Use PRAGMA to get table info
    schema_query = f"PRAGMA table_info({table});"
    schema_df = pd.read_sql_query(schema_query, conn)
    print(schema_df.to_string(index=False))

# ========================================
# 2. HELPER FUNCTION
# ========================================

def show_query(title, query):
    print(f"\n{'='*50}")
    print(f"OPERATION: {title}")
    print('='*50)
    try:
        df = pd.read_sql_query(query, conn)
        print(df.to_string(index=False))
    except Exception as e:
        print(f"--- ERROR IN QUERY ---")
        print(f"Error: {e}")

# ========================================
# 3. RUN OLAP OPERATIONS
# ========================================

# --- 1. ROLL-UP (Summarize by Region) ---
# Changes:
# - 'fact_sales' -> 'sales'
# - 'dim_store' -> 'regions'
# - 'total_amount' -> 'transaction_amount'
# - 'region' -> 'region_name'
query_rollup = """
SELECT region_name, SUM(transaction_amount) AS Total_Revenue
FROM sales
JOIN regions USING (region_id)
GROUP BY region_name
ORDER BY Total_Revenue DESC;
"""
show_query("ROLL-UP: Total Revenue by Region", query_rollup)

# --- 2. DRILL-DOWN (Region → Product) ---
# Note: The 'regions' table has NO 'city' column.
# I changed this query to drill down by 'product_name' instead.
query_drilldown = """
SELECT
    regions.region_name,
    products.product_name,
    SUM(sales.transaction_amount) AS Product_Revenue
FROM sales
JOIN regions USING (region_id)
JOIN products USING (product_id)
GROUP BY regions.region_name, products.product_name
ORDER BY regions.region_name, Product_Revenue DESC;
"""
show_query("DRILL-DOWN: Revenue by Product within Region", query_drilldown)

# --- 3. SLICE (Filter by Quarter Q1) ---
# Note: The 'time' table has NO 'quarter' column.
# I am using 'time_month' to create the 'Q1' slice.
query_slice = """
SELECT
    products.product_name,
    SUM(sales.transaction_amount) AS Q1_Revenue
FROM sales
JOIN products USING (product_id)
JOIN time USING (time_id)
WHERE time.time_month IN (1, 2, 3)  -- This is Q1
GROUP BY products.product_name
ORDER BY Q1_Revenue DESC;
"""
show_query("SLICE: Q1 Sales by Product", query_slice)

# --- 4. DICE (Products in 'West' Region) ---
# Note: The 'products' table has NO 'category' column.
# Note: The 'regions' table has NO 'city' column.
# I removed the 'category' and 'city' filters.
# This query now dices for 'product_name' in the 'region_name' = 'West'
query_dice = """
SELECT
    products.product_name,
    SUM(sales.transaction_amount) AS Revenue
FROM sales
JOIN products USING (product_id)
JOIN regions USING (region_id)
WHERE regions.region_name = 'West'  -- Assumes 'West' exists in your data
GROUP BY products.product_name
ORDER BY Revenue DESC;
"""
show_query("DICE: Product Sales in West Region", query_dice)

# --- 5. PIVOT (Product vs Quarter) ---
# Note: We must build the 'quarter' from 'time_month'.
# I use a WITH clause (Common Table Expression) to do this.
query_pivot = """
WITH SalesWithQuarter AS (
    -- First, create a temporary table with a 'quarter' column
    SELECT
        sales.product_id,
        sales.transaction_amount,
        CASE
            WHEN time.time_month IN (1, 2, 3) THEN 'Q1'
            WHEN time.time_month IN (4, 5, 6) THEN 'Q2'
            WHEN time.time_month IN (7, 8, 9) THEN 'Q3'
            WHEN time.time_month IN (10, 11, 12) THEN 'Q4'
        END AS quarter
    FROM sales
    JOIN time USING (time_id)
)
-- Now, pivot the temporary table
SELECT
  products.product_name,
  SUM(CASE WHEN swq.quarter = 'Q1' THEN swq.transaction_amount ELSE 0 END) AS Q1,
  SUM(CASE WHEN swq.quarter = 'Q2' THEN swq.transaction_amount ELSE 0 END) AS Q2,
  SUM(CASE WHEN swq.quarter = 'Q3' THEN swq.transaction_amount ELSE 0 END) AS Q3,
  SUM(CASE WHEN swq.quarter = 'Q4' THEN swq.transaction_amount ELSE 0 END) AS Q4
FROM SalesWithQuarter AS swq
JOIN products USING (product_id)
GROUP BY products.product_name;
"""
show_query("PIVOT: Product Revenue by Quarter", query_pivot)

print("\n✅ All OLAP Operations Completed Successfully!")
conn.close()