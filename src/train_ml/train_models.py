import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# Load cleaned dataset
# print(os.getcwd())
df = pd.read_csv("src/train_ml/data/cleaned-realtor-data.csv")


print("Loaded cleaned data.")

df = df[:100]  # In local - for faster processing - remove in remote
print("DF Shape: ", df.shape)

# print(df.head())

# Ensure zip_code is formatted as 5-character string with zero-padding
df['zip_code'] = df['zip_code'].astype(int).astype(str).str.zfill(5)

# Preprocessing
X_numeric = df[['bed', 'bath', 'acre_lot', 'house_size']]
# X_categorical = pd.get_dummies(
#     df[['state', 'city', 'zip_code']], drop_first=True)
X_categorical = pd.get_dummies(
    df[['zip_code']], drop_first=True)
X = pd.concat([X_numeric.reset_index(drop=True),
              X_categorical.reset_index(drop=True)], axis=1)
y = df['price']

features = [col for col in X.columns]
print("Features: ", features)
# print(X.columns)
# print(X)
# print(X.dtypes)


print("Preprocessing done.")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Save train data for SHAP
X_train.to_csv("X_train.csv", index=False)

# Define models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}
best_score = -np.inf
best_model = None
best_model_name = ""

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

    results[name] = {
        "R2": r2,
        "RMSE": rmse,
        "CV_R2": cv_r2
    }

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# # Save best model
# os.makedirs("models", exist_ok=True)
# pickle.dump(best_model, open("models/latest_model.pkl", "wb"))
# Save all models individually
os.makedirs("src/train_ml/models", exist_ok=True)
for name, model in models.items():
    filename = f"src/train_ml/models/{name.lower()}_model.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)

# Save features
with open("src/train_ml/models/features.json", "w") as f:
    json.dump(features, f)

# Save metrics
summary = {
    "models": results,
    "best_model": best_model_name,
    "features_used": features,
    "data_size": len(df)
}

with open("metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"✅ Retraining complete. Best model: {best_model_name}")

# 0     103378.0  for_sale  105000.0  3.0   2.0      0.12  1962661.0       Adjuntas  Puerto Rico     601.0       920.0            NaN

# {
#     "bed": 3,
#     "bath": 3,
#     "acre_lot": 0.12,
#     "house_size": 920.0,
#     "zip_code": 601.0
# }
