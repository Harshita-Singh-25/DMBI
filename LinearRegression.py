import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Import Scikit-Learn tools ---
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
# (No 'sns' needed, we are loading data from sklearn)

# ========================================
# CONFIGURATION
# ========================================
# --- For the built-in California Housing dataset ---
TARGET_COLUMN = 'MedHouseVal'  # (This is the median house value)
COLUMNS_TO_DROP = []

# --- FOR YOUR EXAM (if you use a CSV) ---
# DATASET_FILE = 'your_exam_file.csv'  
# TARGET_COLUMN = 'price' # Or 'charges', 'score', etc.
# COLUMNS_TO_DROP = ['Id', 'student_name'] # Any useless columns

# ========================================
# 1. LOAD AND PREPROCESS DATA
# ========================================
print("="*50)
print("LINEAR REGRESSION MODEL: CALIFORNIA HOUSING")
print("="*50)

# --- OPTION A: Load the built-in dataset (NO FILE NEEDED) ---
print("Loading built-in California Housing dataset...")
data = fetch_california_housing()
# Create a single DataFrame (easier to work with)
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name=TARGET_COLUMN)
df = pd.concat([X, y], axis=1)

# --- OPTION B: Load from CSV (for your exam) ---
# Uncomment these lines if you are given a CSV file
# try:
#     df = pd.read_csv(DATASET_FILE)
# except FileNotFoundError:
#     print(f"FATAL ERROR: File not found at {DATASET_FILE}")
#     exit()
# ---

print(f"Dataset shape: {df.shape}\n")

# Drop specified columns
for col in COLUMNS_TO_DROP:
    if col in df.columns:
        print(f"Dropping column: {col}")
        df = df.drop(col, axis=1)

# Separate features and target
y = df[TARGET_COLUMN]
X = df.drop(TARGET_COLUMN, axis=1)

# Handle categorical features (One-Hot Encoding)
# The housing dataset has no categorical data, but this code
# will run for other datasets (like your insurance.csv)
X = pd.get_dummies(X, drop_first=True)

print(f"Features: {X.columns.tolist()}")
print(f"Target: {TARGET_COLUMN}\n")

# ========================================
# 2. HANDLE MISSING VALUES
# ========================================
# The housing dataset has no missing values, but this
# code will run if your exam dataset does.
if X.isnull().sum().any():
    print("Handling missing values...")
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    print("✅ Missing values imputed\n")
else:
    print("No missing values found.\n")

# ========================================
# 3. TRAIN-TEST SPLIT
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples\n")

# ========================================
# 4. BUILD LINEAR REGRESSION MODEL
# ========================================

lr_model = LinearRegression()

print("Training Linear Regression Model...")
lr_model.fit(X_train, y_train)
print("✅ Training completed\n")

# ========================================
# 5. PREDICTIONS
# ========================================

y_pred = lr_model.predict(X_test)

# ========================================
# 6. MODEL EVALUATION
# ========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print("\n--- Interpretation ---")
print(f"The model explains {r2*100:.2f}% of the variance in {TARGET_COLUMN}.")
print(f"On average, the model's predictions are off by {mae:.4f} units (MAE).\n")

# ========================================
# 7. MODEL COEFFICIENTS
# ========================================

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("="*50)
print("MODEL COEFFICIENTS")
print("="*50)
print(coefficients.to_string(index=False))
print(f"\nIntercept: {lr_model.intercept_:.4f}\n")

# ========================================
# 8. ACTUAL VS PREDICTED PLOT
# ========================================

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_actual_vs_predicted.png')
plt.show()
print("✅ Actual vs Predicted plot saved")

# ========================================
# 9. RESIDUALS PLOT
# ========================================

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='k')
plt.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lr_residuals.png')
plt.show()
print("✅ Residual plot saved")

print("\n✅ Linear Regression Model Completed Successfully!")