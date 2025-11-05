import pandas as pd
# import io # No longer needed
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
# 1. LOAD AND PREPROCESS DATA
# ========================================
print("="*50)
print("NAÏVE BAYES CLASSIFICATION (Breast Cancer)")
print("="*50)
print(f"Original dataset shape: {df.shape}\n")

# Drop ID column if exists
if ID_COLUMN and ID_COLUMN in df.columns:
    df = df.drop(ID_COLUMN, axis=1)

# Drop unnamed columns
unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)
    print(f"Dropped columns: {unnamed_cols}\n")

# Separate features and target
y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)

print(f"Features: {X.columns.tolist()}")
print(f"Target: {TARGET_COLUMN}")
print(f"Classes: {y.unique()}\n")

# ========================================
# 2. HANDLE MISSING VALUES
# ========================================

if X.isnull().sum().any():
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("✅ Missing values imputed\n")
else:
    print("No missing values found\n")

# ========================================
# 3. TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# ========================================
# 4. BUILD NAÏVE BAYES MODEL
# ========================================

nb_model = GaussianNB()

print("Training Naïve Bayes Classifier...")
nb_model.fit(X_train, y_train)
print("✅ Training completed\n")

# ========================================
# 5. PREDICTIONS
# ========================================

y_pred = nb_model.predict(X_test)

# ========================================
# 6. MODEL EVALUATION
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
# 7. VISUALIZE CONFUSION MATRIX
# ========================================

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=nb_model.classes_,
            yticklabels=nb_model.classes_)
plt.title('Confusion Matrix - Naïve Bayes')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('nb_confusion_matrix.png')
plt.show()
print("\n✅ Confusion matrix saved")

# ========================================
# 8. PROBABILITY PREDICTIONS
# ========================================

y_proba = nb_model.predict_proba(X_test)
prob_df = pd.DataFrame(y_proba, columns=nb_model.classes_)

# Reset index of y_test to align with prob_df
y_test_reset = y_test.reset_index(drop=True)

prob_df['Actual'] = y_test_reset
prob_df['Predicted'] = y_pred

print("\n" + "="*50)
print("SAMPLE PROBABILITY PREDICTIONS")
print("="*50)
print(prob_df.head(10).to_string(index=False))

# ========================================
# 9. CLASS PROBABILITIES VISUALIZATION
# ========================================

# Plotting probabilities for a subset of samples for clarity
num_samples_to_plot = 20
y_proba_subset = y_proba[:num_samples_to_plot]

plt.figure(figsize=(10, 6))
for i, class_name in enumerate(nb_model.classes_):
    plt.scatter(range(len(y_proba_subset)), y_proba_subset[:, i],
                label=class_name, alpha=0.7, s=100)
plt.xlabel('Sample Index')
plt.ylabel('Probability')
plt.title(f'Prediction Probabilities (First {num_samples_to_plot} test samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('nb_probabilities.png')
plt.show()
print("✅ Probability plot saved")

print("\n✅ Naïve Bayes Classification Completed Successfully!")