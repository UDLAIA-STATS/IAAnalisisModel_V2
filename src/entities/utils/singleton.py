from abc import ABCMeta
import threading


class Singleton(type):
    _instances = {}
    _locks = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._locks:
            cls._locks[cls] = threading.Lock()

        with cls._locks[cls]:
            if cls in cls._instances:
                return cls._instances[cls]

            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            return instance


class AbstractSingleton(Singleton, ABCMeta):
    """Metaclass combining ABC and Singleton safely."""

    pass
