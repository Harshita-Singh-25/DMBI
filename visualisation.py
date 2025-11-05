import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

df = sns.load_dataset('iris')
# Load dataset
#df = pd.read_csv('Iris.csv')  # Change as needed

print("="*50)
print("DATA VISUALIZATION")
print("="*50)

# ========================================
# 1. LINE PLOT
# ========================================

plt.figure(figsize=(10, 5))
numeric_cols = df.select_dtypes(include=np.number).columns[:2]
for col in numeric_cols:
    plt.plot(df.index[:50], df[col][:50], marker='o', label=col)
plt.title('Line Plot - Trend Analysis')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.savefig('viz_lineplot.png')
plt.show()
print("✅ Line plot saved")

# ========================================
# 2. BAR CHART
# ========================================

if 'Species' in df.columns:
    plt.figure(figsize=(8, 5))
    species_counts = df['Species'].value_counts()
    species_counts.plot(kind='bar', color='teal', edgecolor='black')
    plt.title('Bar Chart - Species Distribution')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('viz_barchart.png')
    plt.show()
    print("✅ Bar chart saved")

# ========================================
# 3. HISTOGRAM
# ========================================

plt.figure(figsize=(10, 5))
df.select_dtypes(include=np.number).iloc[:, 0].hist(
    bins=25, color='coral', edgecolor='black', alpha=0.7
)
plt.title('Histogram - Frequency Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)
plt.savefig('viz_histogram.png')
plt.show()
print("✅ Histogram saved")

# ========================================
# 4. SCATTER PLOT
# ========================================

if len(df.select_dtypes(include=np.number).columns) >= 2:
    plt.figure(figsize=(8, 6))
    x_col = df.select_dtypes(include=np.number).columns[0]
    y_col = df.select_dtypes(include=np.number).columns[1]
    
    if 'Species' in df.columns:
        colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
        for species in df['Species'].unique():
            subset = df[df['Species'] == species]
            plt.scatter(subset[x_col], subset[y_col], 
                       label=species, alpha=0.7, s=50)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.7)
    
    plt.title('Scatter Plot - Feature Relationship')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.savefig('viz_scatterplot.png')
    plt.show()
    print("✅ Scatter plot saved")

# ========================================
# 5. BOX PLOT
# ========================================

plt.figure(figsize=(10, 6))
numeric_data = df.select_dtypes(include=np.number).iloc[:, :4]
numeric_data.boxplot()
plt.title('Box Plot - Outlier Detection')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('viz_boxplot.png')
plt.show()
print("✅ Box plot saved")

# ========================================
# 6. HEATMAP
# ========================================

plt.figure(figsize=(8, 6))
correlation = df.select_dtypes(include=np.number).corr()
sns.heatmap(correlation, annot=True, cmap='RdYlGn', 
            center=0, square=True, linewidths=1)
plt.title('Heatmap - Correlation Matrix')
plt.tight_layout()
plt.savefig('viz_heatmap.png')
plt.show()
print(" Heatmap saved")

# ========================================
# 7. PIE CHART
# ========================================

if 'Species' in df.columns:
    plt.figure(figsize=(8, 8))
    species_counts = df['Species'].value_counts()
    plt.pie(species_counts, labels=species_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
    plt.title('Pie Chart - Species Proportion')
    plt.savefig('viz_piechart.png')
    plt.show()
    print(" Pie chart saved")

print("\n All Visualizations Completed Successfully!")
print("Generated files: viz_lineplot.png, viz_barchart.png, viz_histogram.png,")
print("                 viz_scatterplot.png, viz_boxplot.png, viz_heatmap.png, viz_piechart.png")