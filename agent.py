from openai import OpenAI, APIError, RateLimitError
from typing import Callable
from utils import generate_json_schema_for_function
from typing import Literal, Optional
import json
from logger import logger


class OpenAIAgent:
    N_RETRIES = 3

    def __init__(self, system_prompt: str, model: str):
        self.client = OpenAI()
        self.model = model
        self.functions = {}
        self.messages = [{"role": "system", "content": system_prompt}]

    def register_function(self, func: Callable):
        self.functions[func.__name__] = {
            "function": func,
            "schema": generate_json_schema_for_function(func),
        }

    def run_single(
        self, prompt: Optional[str], save_interaction=True, **kwargs
    ) -> Optional[str]:
        """Take a prompt and run it through the client

        Args:
            prompt: The prompt to run through the client
        """

        new_messages = []
        if prompt is not None:
            new_messages.append({"role": "user", "content": prompt})
        logger.debug(f"Running prompt: {prompt}")
        for _ in range(self.N_RETRIES):
            try:
                chat_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages + new_messages,
                    tools=[f["schema"] for f in self.functions.values()],
                    **kwargs,
                )
                break
            except (APIError, TimeoutError, RateLimitError):
                logger.exception("Failed to get chat response. Trying again.")
        message = chat_response.choices[0].message

        new_messages.append(message)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                logger.debug(f"Got tool call response: {function_name}")
                tool_call_id = tool_call.id
                function = self.functions[function_name]["function"]
                try:
                    function_args = json.loads(tool_call.function.arguments)
                    ret = function(**function_args)
                except Exception:
                    new_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": "CALL FAILED: {e}",
                        }
                    )
                    continue

                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": function_name,
                        "content": str(ret),
                    }
                )
        else:
            assert message.content is not None
            logger.debug(f"Got response: {message.content}")
            new_messages.append({"role": "assistant", "content": message.content})
        if save_interaction:
            self.messages.extend(new_messages)
        return message.content

    def run(self, prompt: str, **kwargs) -> str:
        """Take a prompt and run it through the client

        Args:
            prompt: The prompt to run through the client
        """
        while (ret := self.run_single(prompt, **kwargs)) is None:
            prompt = None

        return ret


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
        return "sunny" if "rio" in location.lower() else "bloody terrible"

    agent.register_function(get_current_weather)

    print(agent.run("What is the weather in San Fran?"))
    print(agent.run("What about in Rio?"))

    def calculate_funny_duddy_of_ints(nums: list[int]) -> str:
        """Calculate the funny duddy of a list of integers

        Args:
            nums: The integers to calculate the funny duddy of
        """
        return "Silly sausages"

    agent.register_function(calculate_funny_duddy_of_ints)

    print(agent.run("What is the funny duddy of the digits of 132?"))

    [print(m) for m in agent.messages]
