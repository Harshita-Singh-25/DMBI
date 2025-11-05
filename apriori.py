# ===============================================
# Association Rule Mining (Apriori Algorithm)
# ===============================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
print("Loading Groceries_dataset.csv...")
df = pd.read_csv("Groceries_dataset.csv")

# Create transactions - Group by Member_number and Date
print("Creating transaction data...")
transactions = (
    df.groupby(['Member_number', 'Date'])['itemDescription']
    .apply(list)
    .tolist()  
)

print(f"Total transactions: {len(transactions)}")

# Transaction Encoding
print("Encoding transactions...")
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

print(f"Encoded data shape: {df_encoded.shape}")
print(f"Number of unique items: {len(te.columns_)}")

# Apriori Algorithm
print("\n=== RUNNING APRIORI ALGORITHM ===")
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.02

print(f"Minimum Support: {MIN_SUPPORT}")
print(f"Minimum Confidence: {MIN_CONFIDENCE}")

frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
print(f"Found {len(frequent_itemsets)} frequent itemsets")

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

print(f"Generated {len(rules)} association rules")

# Display Results
print("\n" + "="*50)
print("TOP 5 ASSOCIATION RULES (Sorted by Lift)")
print("="*50)

if not rules.empty:
    # Get top 5 rules
    top_5_rules = rules.head(5)
    
    # Display top 5 rules in clean format
    for i, rule in top_5_rules.iterrows():
        print(f"\n RULE {i+1}:")
        print(f"   IF {set(rule['antecedents'])}")
        print(f"   THEN {set(rule['consequents'])}")
        print(f"    Support: {rule['support']:.3f} | Confidence: {rule['confidence']:.3f} | Lift: {rule['lift']:.3f}")
    
    # Simple Visualization for Top 5 Rules
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Lift values for top 5 rules
    plt.subplot(1, 2, 1)
    plt.barh(range(5), top_5_rules['lift'], color='skyblue', edgecolor='black')
    plt.yticks(range(5), [f"Rule {i+1}" for i in range(5)])
    plt.xlabel('Lift Value')
    plt.title('Top 5 Rules by Lift')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Plot 2: Support vs Confidence scatter for top 5 rules
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(top_5_rules['support'], top_5_rules['confidence'], 
                         s=100, c=top_5_rules['lift'], cmap='viridis', alpha=0.7)
    
    # Annotate points with rule numbers
    for j, (x, y) in enumerate(zip(top_5_rules['support'], top_5_rules['confidence'])):
        plt.annotate(f'Rule {j+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (Top 5 Rules)')
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('top5_apriori_rules.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n Top 5 rules visualization saved as 'top5_apriori_rules.png'")
    
else:
    print("No association rules found. Try lowering min_support or min_confidence.")

print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)

# Basic statistics
transaction_sizes = [len(t) for t in transactions]
print(f" Transaction Analysis:")
print(f"   • Total transactions: {len(transactions)}")
print(f"   • Average items per transaction: {np.mean(transaction_sizes):.2f}")
print(f"   • Unique products: {len(te.columns_)}")

# Most frequent items
item_frequencies = df_encoded.sum().sort_values(ascending=False)
print(f"\n Top 5 Most Popular Items:")
for j, (item, freq) in enumerate(item_frequencies.head(5).items(), 1):
    print(f"   {j}. {item}: {freq} transactions ({freq/len(transactions)*100:.1f}%)")