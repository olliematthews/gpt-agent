from openai import OpenAI
from typing import Callable
from utils import generate_json_schema_for_function
from typing import Literal

import json


class OpenAIAgent:
    def __init__(self, system_prompt: str, model: str):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model = model
        self.functions = {}

    def register_function(self, func: Callable):
        self.functions[func.__name__] = {
            "function": func,
            "schema": generate_json_schema_for_function(func),
        }

    def run(self, prompt: str, **kwargs) -> str:
        """Take a prompt and run it through the client

        Args:
            prompt: The prompt to run through the client
        """

        ret = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tools=[f["schema"] for f in self.functions.values()],
            **kwargs,
        )

        response = ret.choices[0].message
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function = self.functions[function_name]["function"]
                return function(**function_args)
        else:
            assert response.content is not None
            return response.content


if __name__ == "__main__":
    agent = OpenAIAgent("You are a helpful assistant.", "gpt-3.5-turbo-0125")

    def get_current_weather(
        location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"
    ) -> str:
        """Get the current weather

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: The temperature unit to use. Infer this from the users location. Defaults to fahrenheit.
        """
        return "sunny"

    agent.register_function(get_current_weather)

    print(agent.run("What is the weather in San Fran?"))
