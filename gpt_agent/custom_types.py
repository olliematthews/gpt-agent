from typing import Union

JsonType = Union[None, int, str, bool, list["JsonType"], dict[str, "JsonType"]]
