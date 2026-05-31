"""
Ship Profile System & Manual Mapping Configuration.
"""

# The default fallback vessel for manuals not explicitly mapped
DEFAULT_SHIP = "MV_NEPTUNE"

# List of available mock vessels
MOCK_SHIPS = [
    "MV_AURORA",
    "MV_HORIZON",
    "MV_NEPTUNE"
]

# Manual to Ship mapping
SHIP_MANUAL_MAP = {
    "framo-ballast-operation-manual_compress": "MV_AURORA",
    "4d8cc2": "MV_HORIZON",
}

def get_ship_for_manual(manual_name: str) -> str:
    """Returns the assigned ship_id for a given manual name, or the default ship."""
    return SHIP_MANUAL_MAP.get(manual_name, DEFAULT_SHIP)
