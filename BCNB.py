import pandas as pd
# import io # No longer needed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# 0. LOAD DATA
# ========================================
# ---
# NOTE: Please change 'data.csv' to the name of your uploaded file if it's different.
# You must have the file in your Colab session's file system for this to work.
# ---
DATASET_FILE = 'data.csv' 

try:
    df = pd.read_csv(DATASET_FILE)
    print(f"Successfully loaded file: {DATASET_FILE}")
except FileNotFoundError:
    print(f"Error: The file '{DATASET_FILE}' was not found.")
    print("Please make sure you have uploaded the file and the filename matches.")
    # Stop execution if file isn't loaded
    raise
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    raise

# ========================================
# CONFIGURATION
# ========================================
TARGET_COLUMN = 'diagnosis'
ID_COLUMN = 'id'

# ========================================
# 1. PREPROCESS DATA
# ========================================
print("="*50)
print("DECISION TREE CLASSIFICATION (Breast Cancer)")
print("="*50)
print(f"Original dataset shape: {df.shape}\n")

# Drop ID column
if ID_COLUMN in df.columns:
    df = df.drop(ID_COLUMN, axis=1)

# Drop any 'Unnamed' columns that might exist
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

print(f"Features: {X.columns.tolist()}")
print(f"Target: {TARGET_COLUMN}")
print(f"Classes: {y.unique()}\n")

# ========================================
# 2. TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# ========================================
# 3. BUILD DECISION TREE MODEL
# ========================================

dt_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    random_state=42
)

print("Training Decision Tree...")
dt_model.fit(X_train, y_train)
print("✅ Training completed\n")

# ========================================
# 4. PREDICTIONS
# ========================================

y_pred = dt_model.predict(X_test)

# ========================================
# 5. MODEL EVALUATION
# ========================================

accuracy = accuracy_score(y_test, y_pred)
print("="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ========================================
# 6. VISUALIZE CONFUSION MATRIX
# ========================================

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dt_model.classes_,
            yticklabels=dt_model.classes_)
plt.title('Confusion Matrix - Decision Tree')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('dt_confusion_matrix.png')
plt.show()
print("\n✅ Confusion matrix saved")

# ========================================
# 7. VISUALIZE DECISION TREE
# ========================================

plt.figure(figsize=(20, 10))
plot_tree(dt_model,
          feature_names=X.columns.tolist(),
          class_names=dt_model.classes_.astype(str).tolist(),
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Decision tree saved as 'decision_tree.png'")

# ========================================
# 8. FEATURE IMPORTANCE
# ========================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 8)) # Increased height for more features
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('dt_feature_importance.png')
plt.show()

print("\n✅ Decision Tree Classification Completed Successfully!")