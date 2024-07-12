from gpt_agent.utils import generate_json_schema_for_function, json_diff
from enum import Enum
from typing import Literal


class TemperatureUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


EXPECTED_GET_CURRENT_WEATHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location. Defaults to fahrenheit.",
                },
            },
            "required": ["location"],
        },
    },
}


EXPECTED_ADD_NUMS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "add_nums",
        "description": "Add a bunch of numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "nums": {
                    "type": "array",
                    "description": "the numbers to add",
                    "items": {"type": "integer"},
                },
            },
            "required": ["nums"],
        },
    },
}


def test_generate_json_schema_for_function():
    def get_current_weather(
        location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"
    ):
        """Get the current weather

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The temperature unit to use. Infer this from the users location. Defaults to fahrenheit.
        """
        pass

    json_diff(
        generate_json_schema_for_function(get_current_weather),
        EXPECTED_GET_CURRENT_WEATHER_SCHEMA,
    )
    assert (
        generate_json_schema_for_function(get_current_weather)
        == EXPECTED_GET_CURRENT_WEATHER_SCHEMA
    )

    def get_current_weather(
        location: str, unit: TemperatureUnit = TemperatureUnit.FAHRENHEIT
    ):
        """Get the current weather

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The temperature unit to use. Infer this from the users location. Defaults to fahrenheit.
        """
        pass

    assert (
        generate_json_schema_for_function(get_current_weather)
        == EXPECTED_GET_CURRENT_WEATHER_SCHEMA
    )


def test_list_function():
    def add_nums(nums: list[int]):
        """Add a bunch of numbers

        Args:
            nums: the numbers to add
        """
        pass

    print(generate_json_schema_for_function(add_nums))
    json_diff(generate_json_schema_for_function(add_nums), EXPECTED_ADD_NUMS_SCHEMA)
    assert generate_json_schema_for_function(add_nums) == EXPECTED_ADD_NUMS_SCHEMA
