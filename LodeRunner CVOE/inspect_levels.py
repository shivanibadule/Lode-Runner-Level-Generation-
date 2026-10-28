"""STEP 0 - Inspect all processed level files.
Verifies dimensions, unique characters, and calculates per-class frequencies.
"""

from pathlib import Path
from collections import Counter
import constants

def inspect_dataset(levels_dir=constants.LEVELS_DIR):
    level_files = sorted(list(Path(levels_dir).glob("*.txt")))
    print(f"Total level files found: {len(level_files)}")
    
    unique_chars = set()
    irregular_sizes = []
    unknown_chars = set()
    total_tiles = 0
    tile_counter = Counter()
    level_dimensions = set()

    for p in level_files:
        with open(p, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f if line.rstrip("\r\n")]
        
        rows = len(lines)
        cols_set = {len(r) for r in lines}
        if len(cols_set) == 1:
            cols = next(iter(cols_set))
        else:
            cols = f"variable {cols_set}"
        
        level_dimensions.add((rows, cols))
        if rows != constants.ROWS or cols != constants.COLS:
            irregular_sizes.append((p.name, rows, cols))
            
        for r_idx, line in enumerate(lines):
            for c_idx, char in enumerate(line):
                unique_chars.add(char)
                tile_counter[char] += 1
                total_tiles += 1
                if char not in constants.TILES_DICT:
                    unknown_chars.add((p.name, r_idx, c_idx, char))

    print("\n--- LEVEL DIMENSIONS ---")
    for dims in level_dimensions:
        print(f"  Rows x Cols: {dims[0]} x {dims[1]}")
    if irregular_sizes:
        print(f"  WARNING: Irregular level sizes found: {len(irregular_sizes)}")
        for item in irregular_sizes[:5]:
            print(f"    {item[0]}: {item[1]}x{item[2]}")
    else:
        print("  All levels strictly match 22 rows x 32 columns!")

    print("\n--- UNIQUE CHARACTERS ---")
    print(f"  Found characters: {sorted(list(unique_chars))}")
    if unknown_chars:
        print(f"  WARNING: Unknown characters found: {len(unknown_chars)}")
        for u in list(unknown_chars)[:5]:
            print(f"    Level {u[0]} at ({u[1]},{u[2]}): '{u[3]}'")
    else:
        print("  No unknown characters. All match the VGLC legend!")

    print("\n--- PER-CLASS FREQUENCIES ---")
    print(f"  Total tiles across all levels: {total_tiles} ({len(level_files)} * 22 * 32 = {len(level_files)*22*32})")
    print(f"  {'Tile':<6} {'Meaning':<30} {'Count':<10} {'Percentage':<10}")
    print("  " + "-" * 58)
    for tile in constants.TILES:
        tags = constants.TILES_DICT.get(tile, ["unknown"])
        meaning = ", ".join(tags)
        cnt = tile_counter[tile]
        pct = (cnt / total_tiles) * 100 if total_tiles > 0 else 0
        print(f"  '{tile}'    {meaning:<30} {cnt:<10} {pct:6.2f}%")

if __name__ == "__main__":
    inspect_dataset()
