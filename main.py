from gpt_agent import GPTAgent
from enum import Enum
from logger import logger
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


class TemperatureUnit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


def main():
    agent = GPTAgent("You are a helpful assistant.", "gpt-3.5-turbo-0125")

    def get_current_weather(
        location: str, unit: TemperatureUnit = TemperatureUnit.FAHRENHEIT
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

    agent.save_messages_to_file(
        RESULTS_DIR / f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_agent_message.py"
    )


if __name__ == "__main__":
    logger.info("RUNNING...")
    main()
