"""
This has a main agent with multiple workers
"""

from pathlib import Path
import sys
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn
from typing import Optional, Dict, List, Tuple
from openai import OpenAI
import os
import numpy as np
from scipy import stats
import json
from src.train_ml.predict_price import predict_price
from concurrent.futures import ThreadPoolExecutor
from game_sdk.game.worker import Worker
from game_sdk.game.custom_types import Function, Argument, FunctionResult, FunctionResultStatus

# Load .env file
load_dotenv()

# Configuration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Get the absolute path to the project root (where `src` and `game_sdk` live)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))  # Now Python can see `src` and `game_sdk`


game_api_key = os.getenv("GAME_SDK_API_KEY")
if not game_api_key:
    raise ValueError("GAME_SDK_API_KEY is not set")


# MODELS = ["linearregression", "lasso", "ridge", "randomforest", "xgboost"]
MODELS = ["linearregression"]

# FastAPI App
app = FastAPI()


class HouseFeatures(BaseModel):
    bed: float
    bath: float
    acre_lot: float
    house_size: float
    zip_code: float


class PriceEstimationRequest(BaseModel):
    text: str


class PriceEstimationResponse(BaseModel):
    features: dict
    predictions: dict
    aggregated: dict
    justification: str


class FeatureExtractorWorker(Worker):
    def __init__(self):
        super().__init__(
            api_key=game_api_key,
            description="Worker that extracts house features from natural language using OpenAI",
            instruction="Accurately extract bed, bath, acre_lot, house_size, and zip_code from text",
            get_state_fn=self.get_state_fn,
            action_space=[
                Function(
                    fn_name="extract_features",
                    fn_description="Extract house features from text using OpenAI",
                    args=[Argument(name="input_text", type="str",
                                   description="House description")],
                    executable=self.extract_features
                )
            ],
            model_name="Llama-3.1-405B-Instruct"
        )

    def get_state_fn(self, function_result: Optional[FunctionResult], current_state: dict) -> dict:
        if current_state is None:
            return {"last_extraction": None}

        if function_result and hasattr(function_result, 'status') and function_result.status == FunctionResultStatus.DONE:
            if hasattr(function_result, 'info'):
                current_state["last_extraction"] = function_result.info.get(
                    "features")
        return current_state

    def extract_features(self, input_text: str, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
        """Use OpenAI to extract house features from text"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a real estate data extractor. 
                     Extract the following from the text and return ONLY valid JSON:
                     {
                         "bed": number,
                         "bath": number,
                         "acre_lot": number,
                         "house_size": number,
                         "zip_code": number
                     }"""},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            features = json.loads(content.strip())
            HouseFeatures(**features)  # Validate
            return FunctionResultStatus.DONE, "Features extracted successfully", {
                "features": features,
                "input_text": input_text
            }
        except json.JSONDecodeError:
            return FunctionResultStatus.FAILED, "Failed to parse features from OpenAI response", {}
        except Exception as e:
            return FunctionResultStatus.FAILED, f"Feature extraction failed: {str(e)}", {}


class ModelPredictionWorker(Worker):
    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(
            api_key=game_api_key,
            description=f"Worker that makes predictions using {model_name} model",
            instruction=f"Use the {model_name} model to make accurate price predictions",
            get_state_fn=self.get_state_fn,
            action_space=[
                Function(
                    fn_name="predict",
                    fn_description=f"Make price prediction with {model_name}",
                    args=[Argument(name="input_features",
                                   type="dict", description="House features")],
                    executable=self.predict
                )
            ],
            model_name="Llama-3.1-405B-Instruct"
        )

    def get_state_fn(self, function_result: Optional[FunctionResult], current_state: dict) -> dict:
        if current_state is None:
            return {"last_prediction": None}

        if function_result and hasattr(function_result, 'status') and function_result.status == FunctionResultStatus.DONE:
            if hasattr(function_result, 'info'):
                current_state["last_prediction"] = function_result.info.get(
                    "prediction")
        return current_state

    def predict(self, input_features: Dict[str, float], **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
        """Make prediction with this worker's model"""
        try:
            HouseFeatures(**input_features)  # Validate
            price = predict_price(
                bed=input_features["bed"],
                bath=input_features["bath"],
                acre_lot=input_features["acre_lot"],
                house_size=input_features["house_size"],
                zip_code=input_features["zip_code"],
                model=self.model_name
            )
            return FunctionResultStatus.DONE, f"Prediction from {self.model_name}", {
                "prediction": price,
                "model": self.model_name,
                "input_features": input_features
            }
        except Exception as e:
            return FunctionResultStatus.FAILED, f"Prediction failed: {str(e)}", {}


class AggregatorWorker(Worker):
    def __init__(self):
        super().__init__(
            api_key=game_api_key,
            description="Main worker that coordinates feature extraction and prediction aggregation",
            instruction="Extract features, collect predictions from model workers, and aggregate results",
            get_state_fn=self.get_state_fn,
            action_space=[
                Function(
                    fn_name="estimate_price",
                    fn_description="Estimate house price with meta-analysis and justification",
                    args=[Argument(name="input_text", type="str",
                                   description="House description")],
                    executable=self.estimate_price
                )
            ],
            model_name="Llama-3.1-405B-Instruct"
        )
        self.feature_extractor = FeatureExtractorWorker()
        self.model_workers = {
            model: ModelPredictionWorker(model) for model in MODELS}
        self.executor = ThreadPoolExecutor(max_workers=len(MODELS))

    def get_state_fn(self, function_result: Optional[FunctionResult], current_state: dict) -> dict:
        if current_state is None:
            return {
                "last_analysis": None,
                "last_features": None,
                "last_predictions": {}
            }

        if function_result and hasattr(function_result, 'status') and function_result.status == FunctionResultStatus.DONE:
            if hasattr(function_result, 'info'):
                current_state.update({
                    "last_analysis": function_result.info.get("aggregated"),
                    "last_features": function_result.info.get("features"),
                    "last_predictions": function_result.info.get("predictions")
                })
        return current_state

    def remove_outliers(self, predictions: Dict[str, float]) -> Dict[str, float]:
        """Remove outliers using z-score method"""
        values = np.array(list(predictions.values()))
        if len(values) < 3:  # Need at least 3 points for z-score
            return predictions

        z_scores = np.abs(stats.zscore(values))
        return {k: v for k, v in predictions.items()
                if z_scores[list(predictions.keys()).index(k)] < 2}

    def generate_justification(self, features: dict, predictions: dict, aggregated: dict) -> str:
        """Generate justification using OpenAI"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze these house features and price predictions:
                    
                    House Features:
                    - Bedrooms: {features['bed']}
                    - Bathrooms: {features['bath']}
                    - Lot Size (acres): {features['acre_lot']}
                    - House Size (sqft): {features['house_size']}
                    - ZIP Code: {features['zip_code']}
                    
                    Model Predictions:
                    {json.dumps(predictions, indent=2)}
                    
                    Aggregated Results:
                    - Average Price: ${aggregated['average']:,.2f}
                    - Median Price: ${aggregated['median']:,.2f}
                    
                    Provide a 2-3 sentence justification for the predicted price considering:
                    1. The house features
                    2. The model predictions consistency
                    3. Any notable factors affecting the price"""
                }],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Price prediction based on multiple model analyses."

    def estimate_price(self, input_text: str, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
        """Main estimation pipeline"""
        try:
            # Step 1: Extract features
            status, message, info = self.feature_extractor.extract_features(
                input_text)
            if status != FunctionResultStatus.DONE:
                return FunctionResultStatus.FAILED, f"Feature extraction failed: {message}", {}

            features = info["features"]

            # Step 2: Get predictions from all models (parallel)
            def get_prediction(model_name):
                worker = self.model_workers[model_name]
                return worker.predict(features)

            results = list(self.executor.map(get_prediction, MODELS))

            # Step 3: Process predictions
            predictions = {}
            errors = []
            for status, message, info in results:
                if status == FunctionResultStatus.DONE:
                    predictions[info["model"]] = info["prediction"]
                else:
                    errors.append(message)

            if not predictions:
                return FunctionResultStatus.FAILED, "No valid predictions generated. " + "; ".join(errors), {}

            # Step 4: Remove outliers
            filtered_predictions = self.remove_outliers(predictions)

            # Step 5: Aggregate results
            prices = list(filtered_predictions.values())
            aggregated = {
                "average": round(np.mean(prices), 2),
                "median": round(np.median(prices), 2),
                "min": round(min(prices), 2),
                "max": round(max(prices), 2),
                "std_dev": round(np.std(prices), 2),
                "model_count": len(prices),
                "original_count": len(predictions)
            }

            # Step 6: Generate justification
            justification = self.generate_justification(
                features, filtered_predictions, aggregated)

            # Prepare response
            response = {
                "features": features,
                "predictions": {k: round(v, 2) for k, v in filtered_predictions.items()},
                "aggregated": aggregated,
                "justification": justification,
                "input_text": input_text
            }

            return FunctionResultStatus.DONE, "Price estimation complete", response

        except Exception as e:
            return FunctionResultStatus.FAILED, f"Estimation failed: {str(e)}", {}


# Initialize the aggregator worker
aggregator = AggregatorWorker()


@app.post("/estimate", response_model=PriceEstimationResponse)
async def estimate(request: PriceEstimationRequest):
    """API endpoint for price estimation"""
    status, message, info = aggregator.estimate_price(request.text)
    if status != FunctionResultStatus.DONE:
        raise HTTPException(status_code=400, detail=message)
    return info

if __name__ == "__main__":
    # Test the worker implementation
    test_input = "Estimate the price for a house with 3 bed, 3 bath, 0.12 acre lot, 920 house size and with zip code of 601"
    status, message, info = aggregator.estimate_price(test_input)
    if status == FunctionResultStatus.DONE:
        print("Estimation Result:")
        print(json.dumps(info, indent=2))
    else:
        print(f"Error: {message}")

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
