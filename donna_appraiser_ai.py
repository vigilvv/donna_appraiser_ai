"""
Donna real estate appraiser agent
"""

from game_sdk.game.agent import Agent, WorkerConfig
from game_sdk.game.custom_types import Function, Argument, FunctionResult, FunctionResultStatus
from typing import Tuple
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()
game_api_key = os.getenv("GAME_SDK_API_KEY")


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


def get_estimate(num1: float, num2: float, **kwargs) -> Tuple[FunctionResultStatus, str, dict]:
    """
    Function to add two numbers.

    Args:
        num1 (float): First number to add
        num2 (float): Second number to add
        **kwargs: Additional arguments that might be passed

    Returns:
        Tuple[FunctionResultStatus, str, dict]: Status, result message, and additional info
    """
    try:
        result = num1 + num2
        return FunctionResultStatus.DONE, f"The sum of {num1} and {num2} is {result}", {"result": result}
    except Exception as e:
        return FunctionResultStatus.FAILED, f"Failed to add numbers: {str(e)}", {}


# Create the add function
add_numbers_fn = Function(
    fn_name="add",
    fn_description="Add two numbers together",
    args=[
        Argument(name="num1", type="number",
                 description="First number to add"),
        Argument(name="num2", type="number",
                 description="Second number to add")
    ],
    executable=add_numbers
)

# Create a single worker configuration
math_worker = WorkerConfig(
    id="math_worker",
    worker_description="A worker specialized in performing mathematical operations",
    get_state_fn=get_worker_state_fn,
    action_space=[add_numbers_fn]
)

# Create the agent with our math worker
math_agent = Agent(
    api_key=game_api_key,
    name="Calculator",
    agent_goal="Perform mathematical calculations",
    agent_description="You are a simple calculator agent that can perform basic math operations",
    get_agent_state_fn=get_agent_state_fn,
    workers=[math_worker],
    model_name="Llama-3.1-405B-Instruct"
)

# Compile and run the agent
math_agent.compile()
# math_agent.run()

math_agent.get_worker("math_worker").run("Add 5 and 7")
