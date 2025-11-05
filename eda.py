import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

######
#import seaborn as sns
df = sns.load_dataset('iris')
# The seaborn version has different column names, so let's rename one to match your original script's logic (which looks for 'Species').
#df.rename(columns={'species': 'Species'}, inplace=True)
##########

# Load dataset from the QUESTION
#df = pd.read_csv('Iris.csv')  # Change as needed

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# ========================================
# 1. BASIC INFORMATION
# ========================================

print("\n📊 Dataset Overview")
print(f"Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe()}")

# ========================================
# 2. UNIVARIATE ANALYSIS
# ========================================

numeric_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Histograms for numeric features
if len(numeric_cols) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols[:4]):
        axes[i].hist(df[col], bins=20, edgecolor='black', color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('eda_histograms.png')
    plt.show()
    print("\n✅ Histograms saved as 'eda_histograms.png'")

# Count plots for categorical features
if len(categorical_cols) > 0:
    for col in categorical_cols:
        if df[col].nunique() < 20:  # Only plot if reasonable number of categories
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col, palette='viridis')
            plt.title(f'Count of {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'eda_count_{col}.png')
            plt.show()

# ========================================
# 3. BIVARIATE ANALYSIS
# ========================================

# Correlation heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                center=0, fmt='.2f', square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_correlation.png')
    plt.show()
    print("✅ Correlation heatmap saved")

# Pairplot (if target variable exists)
if 'Species' in df.columns:
    sns.pairplot(df, hue='Species', palette='Set2')
    plt.savefig('eda_pairplot.png')
    plt.show()
    print("✅ Pairplot saved")

# ========================================
# 4. OUTLIER DETECTION
# ========================================

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(15, 5))
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(f'{col}')
    axes[i].set_ylabel('Value')

plt.suptitle('Outlier Detection (Boxplots)')
plt.tight_layout()
plt.savefig('eda_boxplots.png')
plt.show()
print("Boxplots saved")

# ========================================
# 5. KEY INSIGHTS
# ========================================

print("\n📈 KEY INSIGHTS:")
print(f"1. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
print(f"2. Numeric features: {len(numeric_cols)}")
print(f"3. Categorical features: {len(categorical_cols)}")

if len(numeric_cols) > 1:
    highest_corr = corr_matrix.abs().unstack()
    highest_corr = highest_corr[highest_corr < 1].sort_values(ascending=False)
    if len(highest_corr) > 0:
        print(f"4. Highest correlation: {highest_corr.index[0]} = {highest_corr.iloc[0]:.3f}")

print("\n✅ EDA Completed Successfully!")