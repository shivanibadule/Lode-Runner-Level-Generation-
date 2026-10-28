"""Constants and Tile Legend definitions for Lode Runner CVAE.
Authoritative tile meanings are loaded dynamically from the VGLC JSON legend.
"""

import json
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
LEGEND_PATH = ROOT_DIR / "Levels" / "Loderunner.json"
LEVELS_DIR = ROOT_DIR / "Levels" / "Processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Grid Dimensions
ROWS = 22
COLS = 32

def load_legend(legend_path=LEGEND_PATH):
    """Load tile legend from JSON and extract tag sets."""
    with open(legend_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tiles_dict = data.get("tiles", {})

    solid = set()
    diggable = set()
    climbable = set()
    passable = set()

    for tile, tags in tiles_dict.items():
        if "solid" in tags:
            solid.add(tile)
        if "diggable" in tags:
            diggable.add(tile)
        if "climbable" in tags:
            climbable.add(tile)
        if "passable" in tags:
            passable.add(tile)

    return tiles_dict, solid, diggable, climbable, passable

TILES_DICT, SOLID, DIGGABLE, CLIMBABLE, PASSABLE = load_legend()

# 8 canonical classes in fixed order
TILES = ['B', 'b', '.', '-', '#', 'G', 'E', 'M']
NUM_CLASSES = len(TILES)
TILE_TO_IDX = {tile: i for i, tile in enumerate(TILES)}
IDX_TO_TILE = {i: tile for i, tile in enumerate(TILES)}

# Open air definition for solver (v1 treats 'E' as open air)
OPEN_AIR = {'.', 'G', 'E', 'M'}

# Standard RGB colors for rendering levels (RGB 0-255)
TILE_COLORS = {
    'B': (130, 70, 30),     # Solid ground / bedrock (Brown)
    'b': (195, 115, 45),    # Diggable brick (Orange-brown)
    '.': (24, 26, 32),      # Passable / empty air (Dark slate)
    '-': (80, 190, 240),    # Rope (Cyan)
    '#': (220, 220, 225),   # Ladder (Silver/White)
    'G': (255, 215, 0),     # Gold (Gold yellow)
    'E': (235, 60, 60),     # Enemy (Red)
    'M': (50, 220, 90),     # Player / Spawn (Vibrant green)
}

# Hex colors for CSS/Streamlit
TILE_COLORS_HEX = {
    tile: f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    for tile, rgb in TILE_COLORS.items()
}
