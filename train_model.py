# train_model.py
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from feature_engineering import add_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# -------------------- Load and preprocess --------------------
df = pd.read_csv("data/historical_data.csv")
df = add_features(df)

# Drop columns safely
cols_to_drop = [c for c in ["time", "Target"] if c in df.columns]
X = df.drop(columns=cols_to_drop)

# Ensure 'Target' exists
if "Target" not in df.columns:
    raise ValueError("❌ 'Target' column not found in historical_data.csv after feature engineering")

y = df["Target"]

# Handle missing or infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(method="ffill", inplace=True)  # forward fill missing values
X.fillna(method="bfill", inplace=True)  # backfill if any remain

# -------------------- Train/test split --------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# -------------------- Train model --------------------
params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "eta": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42
}

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "eval")], verbose_eval=10)

# -------------------- Evaluate --------------------
y_pred_prob = model.predict(dtest)
y_pred = (y_pred_prob > 0.5).astype(int)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Logloss: {log_loss(y_test, y_pred_prob):.4f}")

# -------------------- Save model --------------------
joblib.dump(model, "model/xgb_model.joblib")
print("✅ Model trained and saved")
