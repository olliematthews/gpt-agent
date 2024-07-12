from openai import OpenAI, APIError, RateLimitError
from typing import Callable
from pathlib import Path
from gpt_agent.utils import generate_json_schema_for_function
from typing import Optional
import json
from logger import logger


class GPTAgent:
    N_RETRIES = 3

    def __init__(self, system_prompt: str, model: str):
        self.client = OpenAI()
        self.model = model
        self.functions = {}
        self.message_history = [{"role": "system", "content": system_prompt}]

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
                    messages=self.message_history + new_messages,
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
            self.message_history.extend(new_messages)
        return message.content

    def run(self, prompt: str, **kwargs) -> str:
        """Take a prompt and run it through the client

        Args:
            prompt: The prompt to run through the client
        """
        while (ret := self.run_single(prompt, **kwargs)) is None:
            prompt = None

        return ret

    def save_messages_to_file(self, file_path: Path):
        """Print the message history for this agent"""
        with open(file_path, "w") as fd:
            logger.info(f"Writing messages to {file_path}")
            for message in self.message_history:
                fd.write("----------------------------\n")
                if isinstance(message, dict):
                    match message["role"]:
                        case "system":
                            fd.write("SYSTEM MESSAGE\n")
                        case "user":
                            fd.write("USER MESSAGE\n")
                        case "assistant":
                            fd.write("USER MESSAGE\n")
                        case "tool":
                            fd.write("TOOL CALL\n")
                            fd.write(f"{message['name']}\n")
                    fd.write("\n")
                    fd.write(message["content"])
                    fd.write("\n")
                else:
                    assert (
                        message.role == "assistant"
                    ), "Only implemented expect assistant messages in this format"
                    fd.write("ASSISTANT MESSAGE\n")
                    fd.write("\n")
                    if not message.tool_calls:
                        fd.write(message.content)
                    else:
                        [fd.write(str(tc)) for tc in message.tool_calls]
                    fd.write("\n")
