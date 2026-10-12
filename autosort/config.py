"""Shared configuration values for the AutoSort system."""

BIN_ANGLES = {
    "recycling": 0.0,
    "battery": 0.5,
    "compost": 1,
    "landfill": 1
}

BIN_NAMES = {
    "recycling": "recycling",
    "battery": "battery",
    "compost": "compost",
    "landfill": "landfill",
}

DEFAULT_BIN = "recycling"

LABEL_ALIASES = {
    "recycle": "recycling",
    "recycling": "recycling",
    "battery": "battery",
    "compost": "compost",
    "landfill": "landfill",
    "background_class83422": None,
}

SORTING_RULES = {
    "can": "recycling",
    "bottle": "recycling",
    "plastic": "recycling",
    "recycle": "recycling",
    "recycling": "recycling",
    "battery": "battery",
    "cell": "battery",
    "food": "compost",
    "organics": "compost",
    "compost": "compost",
    "metal": "recycling",
    "aluminum": "recycling",
    "steel": "recycling",
}

SLOT_MAP = {
    "recycling": 1,
    "battery": 2,
    "compost": 3,
    "landfill": 4,
}
