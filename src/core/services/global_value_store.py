from typing import Any, Dict

from typing_extensions import Self


class GlobalValueStore:
    _instance = None

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = {}

        return cls._instance

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key: str, default: Any = None):
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def update(self, data: Dict[str, Any]):
        self._data.update(data)

    def reset(self):
        self._data.clear()


value_store = GlobalValueStore()
