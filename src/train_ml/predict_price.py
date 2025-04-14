import json
import pandas as pd
import pickle

# 1. Load the saved feature list
with open("src/train_ml/models/features.json", "r") as f:
    model_features = json.load(f)


def predict_price(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, model: str):
    # model = lasso, linearregression, randomforest, ridge, xgboost

    zip_code_padded = f"{zip_code:05.0f}"

    input_data = {"bed": bed, "bath": bath, "acre_lot": acre_lot,
                  "house_size": house_size, "zip_code": zip_code_padded}

    # 2. Convert input dict to DataFrame
    # input_data = {
    #     "bed": 3,
    #     "bath": 3,
    #     "acre_lot": 0.12,
    #     "house_size": 920.0,
    #     "zip_code": 601.0
    # }

    # Convert to DataFrame
    X_input = pd.DataFrame([input_data])

    # Dummy encode
    X_input = pd.get_dummies(X_input)

    print(X_input)

    # Align columns with training features
    for col in model_features:
        if col not in X_input.columns:
            X_input[col] = 0  # Fill missing dummy variables

    # Ensure correct order
    X_input = X_input[model_features]

    # 3. Use model to predict
    with open(f"src/train_ml/models/{model}_model.pkl", "rb") as f:
        model = pickle.load(f)

    predicted_price = model.predict(X_input)[0]
    print(f"Predicted price: ${predicted_price:,.2f}")
    return predicted_price
