"""Core AutoSort planning and configuration modules."""

from . import config as config
from .config import BIN_ANGLES, BIN_NAMES, DEFAULT_BIN

__all__ = [
    "config",
    "BIN_ANGLES",
    "BIN_NAMES",
    "DEFAULT_BIN",
    "AutoSortSystem",
    "BridgeRPC",
    "SortAction",
]


def __getattr__(name):
    if name in {"AutoSortSystem", "BridgeRPC", "SortAction"}:
        from .planner import AutoSortSystem, BridgeRPC, SortAction
        mapping = {
            "AutoSortSystem": AutoSortSystem,
            "BridgeRPC": BridgeRPC,
            "SortAction": SortAction,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
