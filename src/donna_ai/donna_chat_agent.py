"""
This follows a chat agent pattern
"""

import os
from typing import Any, Tuple
from dotenv import load_dotenv
from game_sdk.game.chat_agent import ChatAgent
from game_sdk.game.custom_types import Argument, Function, FunctionResultStatus
from src.train_ml.predict_price import predict_price
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

# ACTION SPACE


# def check_crypto_price(currency: str):
#     prices = {
#         "bitcoin": 100000,
#         "ethereum": 20000,
#     }
#     result = prices[currency.lower()]
#     if result is None:
#         return FunctionResultStatus.FAILED, "The price of the currency is not available", {}
#     return FunctionResultStatus.DONE, f"The price of {currency} is {result}", {}


def predict_price_linear_regression(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "linearregression")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to predict price: {str(e)}", {}


def predict_price_lasso(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "lasso")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to predict price: {str(e)}", {}


def predict_price_ridge(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "ridge")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to predict price: {str(e)}", {}


def predict_price_randomforest(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "randomforest")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to predict price: {str(e)}", {}


def predict_price_xgboost(bed: float, bath: float, acre_lot: float, house_size: float, zip_code: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    try:
        result = predict_price(bed, bath, acre_lot,
                               house_size, zip_code, "xgboost")
        return FunctionResultStatus.DONE, f"The predicted price is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to predict price: {str(e)}", {}


action_space = [
    # Function(
    #     fn_name="predict_price_linear_regression",
    #     fn_description="Predict price based on linear regression",
    #     args=[
    #         Argument(name="bed", type="float",
    #                  description="Number of beds"),
    #         Argument(name="bath", type="float",
    #                  description="Number of baths"),
    #         Argument(name="acre_lot", type="float",
    #                  description="Acre lot"),
    #         Argument(name="house_size", type="float",
    #                  description="House size"),
    #         Argument(name="zip_code", type="float",
    #                  description="Zip code"),
    #     ],
    #     executable=predict_price_linear_regression
    # ),
    Function(
        fn_name="predict_price_lasso",
        fn_description="Predict price based on lasso regression",
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
        executable=predict_price_lasso
    ),
    Function(
        fn_name="predict_price_ridge",
        fn_description="Predict price based on ridge regression",
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
        executable=predict_price_ridge
    ),
    Function(
        fn_name="predict_price_random_forest",
        fn_description="Predict price based on random forest",
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
        executable=predict_price_randomforest
    ),
    Function(
        fn_name="predict_price_xgboost",
        fn_description="Predict price based on xgboost",
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
        executable=predict_price_xgboost
    )
]


# CREATE AGENT
agent = ChatAgent(
    prompt="You are master real estate appraiser. Based on the user input you get appraisals from workers, do meta analysis and output a coheisive price to the user in dollars. You call each worker only exactly once.",
    api_key=game_api_key
)

chat = agent.create_chat(
    partner_id="donna",
    partner_name="Donna",
    action_space=action_space,
)

chat_continue = True
while chat_continue:

    user_message = input("Enter a message: ")

    response = chat.next(user_message)
    print("==================")
    print(response)

    if response.function_call:
        print(f"Function call: {response.function_call.fn_name}")

    if response.message:
        print(f"Response: {response.message}")

    if response.is_finished:
        chat_continue = False
        break

print("Chat ended")
