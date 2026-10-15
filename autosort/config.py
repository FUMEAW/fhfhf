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

# ---------------------------------------------------------------------------
# Platform geometry (see "Platform Drawing.png"), top-down view, degrees
# clockwise with 0 at the top of the drawing.
#
# Bins are FIXED under the platform at the four corners. At the home yaw
# position each slot sits directly over the bin in the same corner.
# Rotating the platform carries a slot to a different bin.
# ---------------------------------------------------------------------------

BIN_DIRECTIONS = {
    "recycling": 315,
    "battery": 45,
    "compost": 135,
    "landfill": 225,
}

SLOT_HOME_DIRECTIONS = {
    1: 315,  # over recycling at home
    2: 45,   # over battery at home
    4: 135,  # over compost at home
    3: 225,  # over landfill at home
}

TURNS = (0, 90, 180, 270)


def turn_for(slot: int, bin_name: str) -> int:
    """Clockwise platform turn (0/90/180/270) that puts `slot` over `bin_name`.

    +90 means slot 1 moves from the recycling corner to the battery corner.
    """
    return (BIN_DIRECTIONS[bin_name] - SLOT_HOME_DIRECTIONS[slot]) % 360


def home_slot_for_bin(bin_name: str) -> int:
    """The slot that sits over `bin_name` when the platform is at home."""
    for slot, direction in SLOT_HOME_DIRECTIONS.items():
        if direction == BIN_DIRECTIONS[bin_name]:
            return slot
    raise KeyError(bin_name)
