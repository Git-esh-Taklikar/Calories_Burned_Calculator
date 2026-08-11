import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Import YOUR custom class from your local file
from linear_regression import LinearRegression

# 2. Import Sklearn's version but rename it as SklearnLR so they don't collide!
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# STEP 1: LOAD & CLEAN DATA
# ============================================================
exercise_df = pd.read_csv("data/exercise.csv")
calories_df = pd.read_csv("data/calories.csv")

df = pd.merge(exercise_df, calories_df, on="User_ID")

print(df.shape)
print(df.head())
print(df.describe())

# Encode Gender as numbers
df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

# Drop User_ID — it's not a feature
df = df.drop("User_ID", axis=1)

# Print which features correlate most with Calories
print("\n=== CORRELATIONS WITH CALORIES ===")
print(df.corr()["Calories"].sort_values(ascending=False))


# ============================================================
# STEP 2: PREPARE DATA (The missing piece!)
# ============================================================
# Extract raw matrix values for our top 3 features, and target vector y
X = df[["Duration", "Heart_Rate", "Body_Temp"]].values
y = df["Calories"].values

# Z-Score Normalization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Shuffle data cleanly
np.random.seed(42)
shuffle_idx = np.random.permutation(len(X_norm))
X_norm = X_norm[shuffle_idx]
y = y[shuffle_idx]

# Split 80% train, 20% test
split_idx = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTrain shape: {X_train.shape[0]} samples")
print(f"Test shape:  {X_test.shape[0]} samples")


# ============================================================
# STEP 3: TRAIN YOUR FROM-SCRATCH MODEL
# ============================================================
print("\n" + "=" * 50)
print("FROM-SCRATCH LINEAR REGRESSION")
print("=" * 50)

# Instantiate and fit your custom model
model = LinearRegression(learning_rate=0.01, n_iterations=2000)
model.fit(X_train, y_train)

# Evaluate on training data
train_r2 = model.r2_score(X_train, y_train)
train_mse = model.mse(X_train, y_train)
print(f"Train R²: {train_r2:.4f}")
print(f"Train MSE: {train_mse:.2f}")

# Evaluate on test data
test_r2 = model.r2_score(X_test, y_test)
test_mse = model.mse(X_test, y_test)
print(f"Test R²:  {test_r2:.4f}")
print(f"Test MSE:  {test_mse:.2f}")

print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")


# ============================================================
# STEP 4: SKLEARN COMPARISON
# ============================================================
print("\n" + "=" * 50)
print("SKLEARN LINEAR REGRESSION")
print("=" * 50)

# Instantiate and fit scikit-learn's version
sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)

# Predict using the sklearn model
sklearn_pred = sklearn_model.predict(X_test)

# Calculate metrics using sklearn's built-in functions
sk_r2 = r2_score(y_test, sklearn_pred)
sk_mse = mean_squared_error(y_test, sklearn_pred)

print(f"Weights: {sklearn_model.coef_}")
print(f"Bias: {sklearn_model.intercept_:.4f}")


# ============================================================
# COMPARISON DISPLAY Table
# ============================================================
print("\n" + "=" * 50)
print("COMPARISON")
print("=" * 50)
print(f"{'Metric':<10} {'From Scratch':>15} {'Sklearn':>15}")
print("-" * 40)
print(f"{'R²':<10} {test_r2:>15.4f} {sk_r2:>15.4f}")
print(f"{'MSE':<10} {test_mse:>15.2f} {sk_mse:>15.2f}")
# ============================================================
# STEP 5: FEATURE EXPERIMENTS
# ============================================================
print("\n" + "=" * 50)
print("FEATURE EXPERIMENTS")
print("=" * 50)

feature_sets = {
    "Duration only": ["Duration"],
    "Heart_Rate only": ["Heart_Rate"],
    "Duration + Heart_Rate": ["Duration", "Heart_Rate"],
    "Top 3 features": ["Duration", "Heart_Rate", "Body_Temp"],
    "All features": ["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
}

results = {}

for name, features in feature_sets.items():
    # Extract features and target
    X_exp = df[features].values
    y_exp = df["Calories"].values
    
    # Normalize
    X_mean_exp = X_exp.mean(axis=0)
    X_std_exp = X_exp.std(axis=0)
    # Handle case where a feature has zero variance to avoid division by zero
    X_std_exp[X_std_exp == 0] = 1.0 
    X_norm_exp = (X_exp - X_mean_exp) / X_std_exp
    
    # Shuffle (Using the same seed for fair comparison)
    np.random.seed(42)
    shuffle_idx_exp = np.random.permutation(len(X_norm_exp))
    X_norm_exp = X_norm_exp[shuffle_idx_exp]
    y_exp = y_exp[shuffle_idx_exp]
    
    # Split 80/20
    split_idx_exp = int(0.8 * len(X_norm_exp))
    X_train_exp, X_test_exp = X_norm_exp[:split_idx_exp], X_norm_exp[split_idx_exp:]
    y_train_exp, y_test_exp = y_exp[:split_idx_exp], y_exp[split_idx_exp:]
    
    # Train custom model
    exp_model = LinearRegression(learning_rate=0.01, n_iterations=2000)
    exp_model.fit(X_train_exp, y_train_exp)
    
    # Evaluate
    results[name] = exp_model.r2_score(X_test_exp, y_test_exp)

print(f"\n{'Features Used':<25} {'Test R²':>10}")
print("-" * 38)
for name, r2 in results.items():
    print(f"{name:<25} {r2:>10.4f}")


# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Calorie Burn Predictor — Results", fontsize=16, fontweight="bold")

# Plot 1: Cost vs Iterations
axes[0, 0].plot(model.cost_history, color='blue')
axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("Cost")
axes[0, 0].set_title("Training Convergence")

# Plot 2: Predicted vs Actual
y_pred = model.predict(X_test)
axes[0, 1].scatter(y_test, y_pred, alpha=0.3, s=10, color='purple')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0, 1].set_xlabel("Actual Calories")
axes[0, 1].set_ylabel("Predicted Calories")
axes[0, 1].set_title(f"Predicted vs Actual (R²={test_r2:.4f})")

# Plot 3: Feature Correlations
correlations = df.corr()["Calories"].drop("Calories").sort_values()
axes[0, 2].barh(correlations.index, correlations.values, color='green')
axes[0, 2].set_title("Feature Correlations with Calories")

# Plot 4: Residuals
residuals = y_test - y_pred
axes[1, 0].scatter(y_pred, residuals, alpha=0.3, s=10, color='orange')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel("Predicted Calories")
axes[1, 0].set_ylabel("Residual (Actual - Pred)")
axes[1, 0].set_title("Residuals Plot")

# Plot 5: Duration vs Calories
axes[1, 1].scatter(df["Duration"], df["Calories"], alpha=0.2, s=5, color='darkblue')
axes[1, 1].set_xlabel("Duration (min)")
axes[1, 1].set_ylabel("Calories")
axes[1, 1].set_title("Duration vs Calories")

# Plot 6: Feature Experiment Results
axes[1, 2].bar(results.keys(), results.values(), color='teal')
axes[1, 2].set_ylabel("R² Score")
axes[1, 2].set_title("Feature Selection Impact")
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
print("\n✅ All dashboard plots compiled and saved to results.png")
plt.show()