import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

######
import seaborn as sns
df = sns.load_dataset('iris')
# The seaborn version has different column names, so let's rename one to match your original script's logic (which looks for 'Species').
df.rename(columns={'species': 'Species'}, inplace=True)
##########


# Load dataset stored in the Files in the left side bar:
#df = pd.read_csv('Iris.csv')  # Change filename as needed

print("="*50)
print("ORIGINAL DATA")
print("="*50)
print(f"Shape: {df.shape}")
print(df.head())
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ========================================
# 1. HANDLE MISSING VALUES
# ========================================

# Drop rows where all values are missing
df = df.dropna(how='all')

# For numeric columns: fill with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# For categorical columns: fill with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print("\n Missing values handled")

# ========================================
# 2. REMOVE DUPLICATES
# ========================================

original_rows = len(df)
df = df.drop_duplicates()
print(f" Removed {original_rows - len(df)} duplicate rows")

# ========================================
# 3. HANDLE OUTLIERS (IQR Method)
# ========================================

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply to numeric columns (skip ID columns)
for col in numeric_cols:
    if col not in ['Id', 'id', 'ID']:
        before = len(df)
        df = remove_outliers(df, col)
        removed = before - len(df)
        if removed > 0:
            print(f" Removed {removed} outliers from {col}")

# ========================================
# 4. ENCODE CATEGORICAL VARIABLES
# ========================================

le = LabelEncoder()
for col in categorical_cols:
    if col != 'Id':  # Skip ID columns
        df[col + '_Encoded'] = le.fit_transform(df[col])
        print(f"Encoded {col}")

# ========================================
# 5. FEATURE SCALING (Standardization)
# ========================================

# Select numeric features for scaling (exclude encoded columns)
features_to_scale = [col for col in numeric_cols if col not in ['Id', 'id', 'ID']]

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\nFeature scaling completed")

# ========================================
# FINAL OUTPUT
# ========================================

print("\n" + "="*50)
print("PREPROCESSED DATA")
print("="*50)
print(f"Shape: {df.shape}")
print(df.head())
print(f"\nData Types:\n{df.dtypes}")

print("\nData Preprocessing Completed Successfully!")