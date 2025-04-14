"""
Donna real estate appraiser agent
"""


from src.train_ml.predict_price import predict_price
import os
from dotenv import load_dotenv
from typing import Tuple
from game_sdk.game.custom_types import Function, Argument, FunctionResult, FunctionResultStatus
from game_sdk.game.agent import Agent, WorkerConfig

import sys
from pathlib import Path

# Get the absolute path to the project root (where `src` and `game_sdk` live)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))  # Now Python can see `src` and `game_sdk`


# Load .env file
load_dotenv()
game_api_key = os.getenv("GAME_SDK_API_KEY")
if not game_api_key:
    raise ValueError("GAME_SDK_API_KEY is not set")


def get_worker_state_fn(function_result: FunctionResult, current_state: dict) -> dict:
    """
    Simple state management function for the worker.
    In this example, we'll just maintain a count of operations performed.
    """
    if current_state is None:
        return {"operation_count": 0}

    # Increment the operation count each time a function is called
    new_state = current_state.copy()
    new_state["operation_count"] = current_state.get("operation_count", 0) + 1
    return new_state


def get_agent_state_fn(function_result: FunctionResult, current_state: dict) -> dict:
    """
    Simple state management function for the agent.
    We'll just mirror the worker's state in this simple example.
    """
    if current_state is None:
        return {"operation_count": 0}

    new_state = current_state.copy()
    new_state["operation_count"] = current_state.get("operation_count", 0) + 1
    return new_state


def predict_price_linear_regression(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    """
    Function to predict price based on linear regression.

    Args:
        bed (float): Number of beds
        bath (float): Number of baths
        acre_lot (float): Acre lot
        house_size (float): House size
        zip_code (float): Zip code
        **kwargs: Additional arguments that might be passed

    Returns:
        Tuple[FunctionResultStatus, str, dict]: Status, result message, and additional info
    """
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "linearregression")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to add numbers: {str(e)}", {}


# Create the add function
predict_price_linear_regression_fn = Function(
    fn_name="predict_price_linear_regression",
    fn_description="Predict price based on linear regression",
    args=[
        Argument(name="bed", type="float",
                 description="Number of beds"),
        Argument(name="bath", type="float",
                 description="Number of baths"),
        Argument(name="acre_lot", type="float",
                 description="Acre lot"),
        Argument(name="house_size", type="float",
                 description="House size"),
        Argument(name="zip_code", type="float",
                 description="Zip code"),
    ],
    executable=predict_price_linear_regression
)

# Create a single worker configuration
linear_regression_prediction_worker = WorkerConfig(
    id="linear_regression_prediction_worker",
    worker_description="A worker specialized in predicting real estate price based on the linear regression model",
    get_state_fn=get_worker_state_fn,
    action_space=[predict_price_linear_regression_fn]
)

# Create the agent with our math worker
donna_agent = Agent(
    api_key=game_api_key,
    name="Appraiser",
    agent_goal="You perform detailed real estate appraisals.",
    agent_description="You are master real estate appraiser. Based on the user input you get appraisals from workers, do meta analysis and output a coheisive price to the user in dollars. You call all worker exactly once and then stop.",
    get_agent_state_fn=get_agent_state_fn,
    workers=[linear_regression_prediction_worker],
    model_name="Llama-3.1-405B-Instruct"
)

# Compile and run the agent
donna_agent.compile()
donna_agent.run()

donna_agent.get_worker("linear_regression_prediction_worker").run(
    "Estimate the price for a house with 3 bed, 3 bath, 0.12 acre lot, 920 house size and with zip code of 601")

# input_data = {
#     "bed": 3,
#     "bath": 3,
#     "acre_lot": 0.12,
#     "house_size": 920.0,
#     "zip_code": 601.0
# }
