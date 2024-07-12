"""Helper functions for the GPT Agent."""

import inspect
from typing import Callable, Literal, get_origin, get_args
from docstring_parser import parse
from enum import EnumMeta
from gpt_agent.custom_types import JsonType
from deepdiff import DeepDiff
from pprint import pprint

# Map python types to typeschema type descriptions
TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def json_diff(json1: JsonType, json2: JsonType):
    """
    Print  a string representation of the difference between two JSON objects.

    Args:
        json1: The first JSON object.
        json2: The second JSON object.

    """
    diff = DeepDiff(json1, json2)
    pprint(diff, indent=2)


# Assuming TYPE_MAP and JsonType are defined elsewhere


def expand_type(type_: type) -> dict[str, JsonType]:
    """Expand a type into a JSON schema.

    Args:
        type_: the type to expand out

    Returns:
        the JSON schema for the type
    """
    if type_ in TYPE_MAP:
        return {"type": TYPE_MAP[type_]}
    elif get_origin(type_) == Literal:
        possible_args = list(get_args(type_))

        arg_types = set(type(arg) for arg in possible_args)
        assert len(arg_types) == 1, "All enum values must be of the same type"

        ret = expand_type(arg_types.pop())
        ret.update({"enum": possible_args})  # Add the possible values as an enum
        return ret
    elif isinstance(type_, EnumMeta):
        possible_args = [arg.value for arg in type_]

        arg_types = set(type(arg) for arg in possible_args)
        assert len(arg_types) == 1, "All enum values must be of the same type"

        ret = expand_type(arg_types.pop())
        ret.update({"enum": possible_args})  # Add the possible values as an enum
        return ret
    elif get_origin(type_) is list:
        element_type = get_args(type_)[0]
        item_type = expand_type(element_type)
        return {"type": "array", "items": item_type}
    else:
        raise NotImplementedError(f"Unsupported type: {type_}")


def generate_json_schema_for_function(func: Callable) -> JsonType:
    """
    Generates a JSON schema for a given function.

    Args:
        func: The function to generate a schema for.

    Returns:
        A JSON schema as a list of dictionaries.
    """
    if not callable(func):
        raise ValueError("Provided argument is not a callable function.")

    func_signature = inspect.signature(func)
    func_doc = inspect.getdoc(func) or ""
    parsed_docstring = parse(func_doc)
    schema = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": parsed_docstring.short_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }

    for docstring_arg in parsed_docstring.params:
        param_name = docstring_arg.arg_name
        param = func_signature.parameters[param_name]
        # Assuming all parameters without default values are required
        if param.default is inspect.Parameter.empty:
            schema["function"]["parameters"]["required"].append(param_name)

        param_type = func_signature.parameters[param_name].annotation

        param_properties = expand_type(param_type)
        param_properties["description"] = docstring_arg.description

        schema["function"]["parameters"]["properties"][param_name] = param_properties
    return schema
