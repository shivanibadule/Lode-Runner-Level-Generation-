"""Retro arcade sprite renderer for Lode Runner levels.
Generates authentic pixel-art tiles (24x24) matching classic Lode Runner arcade / NES visuals:
- Red running-bond bricks with mortar joints
- Beveled dark bedrock blocks
- Classic ladder rails with rungs
- Taut horizontal cable rope
- Golden treasure chests with highlights
- Retro pixel hero runner and enemy monk guards
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict, Tuple, Optional
import constants


TILE_SIZE = 24  # 24x24 pixels per tile -> 528 x 768 full HD level canvas


def create_tile_sprites(size: int = TILE_SIZE) -> Dict[str, np.ndarray]:
    """Generate high-resolution pixel art templates for all 8 Lode Runner tiles."""
    sprites = {}

    # 1. Passable Air ('.')
    air = np.zeros((size, size, 3), dtype=np.uint8) # Pitch black
    sprites["."] = air

    # 2. Diggable Brick ('b') - Classic red running-bond brick with black mortar
    brick = np.full((size, size, 3), (180, 32, 24), dtype=np.uint8) # Red brick base
    # Mortar lines (dark grey/black)
    mortar_color = (20, 20, 22)
    highlight_color = (235, 75, 60)
    shadow_color = (110, 18, 12)

    # 3 horizontal courses of bricks
    h_lines = [0, size // 3, (2 * size) // 3, size - 1]
    for y in h_lines:
        brick[y, :, :] = mortar_color

    # Vertical joints (staggered)
    mid_y1 = size // 3
    mid_y2 = (2 * size) // 3

    # Row 1 joints
    brick[0:mid_y1, 0, :] = mortar_color
    brick[0:mid_y1, size // 2, :] = mortar_color
    brick[0:mid_y1, -1, :] = mortar_color

    # Row 2 joints (staggered by 1/4)
    v_stagger1 = size // 4
    v_stagger2 = (3 * size) // 4
    brick[mid_y1:mid_y2, v_stagger1, :] = mortar_color
    brick[mid_y1:mid_y2, v_stagger2, :] = mortar_color

    # Row 3 joints
    brick[mid_y2:size, 0, :] = mortar_color
    brick[mid_y2:size, size // 2, :] = mortar_color
    brick[mid_y2:size, -1, :] = mortar_color

    # Brick bevel highlights
    for y in [1, mid_y1 + 1, mid_y2 + 1]:
        if y < size:
            brick[y, 1:-1, :] = np.clip(brick[y, 1:-1, :] * 1.25, 0, 255)

    sprites["b"] = brick

    # 3. Solid Bedrock ('B') - Heavy metallic/stone block with beveled border and rivets
    bedrock = np.full((size, size, 3), (60, 65, 75), dtype=np.uint8)
    # Bevel top/left bright
    bedrock[0:2, :, :] = (130, 140, 155)
    bedrock[:, 0:2, :] = (130, 140, 155)
    # Bevel bottom/right dark
    bedrock[-2:, :, :] = (25, 28, 35)
    bedrock[:, -2:, :] = (25, 28, 35)
    # Crosshatch / steel rivet corners
    for ry, rx in [(4, 4), (4, size - 5), (size - 5, 4), (size - 5, size - 5)]:
        bedrock[ry : ry + 2, rx : rx + 2, :] = (190, 200, 215)
        bedrock[ry + 2, rx : rx + 2, :] = (20, 20, 25)
    sprites["B"] = bedrock

    # 4. Ladder ('#') - White/silver side rails and 3 horizontal rungs
    ladder = np.zeros((size, size, 3), dtype=np.uint8)
    rail_w = max(2, size // 8)
    rail_l = size // 5
    rail_r = size - rail_l - rail_w

    # Left and Right vertical rails
    ladder[:, rail_l : rail_l + rail_w, :] = (225, 230, 240)
    ladder[:, rail_l, :] = (255, 255, 255) # Highlight
    ladder[:, rail_r : rail_r + rail_w, :] = (225, 230, 240)
    ladder[:, rail_r, :] = (255, 255, 255)

    # 3 Horizontal rungs
    rung_h = max(2, size // 10)
    for ry in [size // 6, size // 2, (5 * size) // 6]:
        ladder[ry : ry + rung_h, rail_l + rail_w : rail_r, :] = (220, 225, 235)
        ladder[ry, rail_l + rail_w : rail_r, :] = (255, 255, 255)

    sprites["#"] = ladder

    # 5. Rope ('-') - High taut horizontal rope with hanging knots
    rope = np.zeros((size, size, 3), dtype=np.uint8)
    rope_y = size // 4
    rope_thick = max(2, size // 10)
    rope[rope_y : rope_y + rope_thick, :, :] = (80, 200, 255) # Cyan rope
    rope[rope_y, :, :] = (180, 240, 255) # Shimmer
    # Periodic knot knots
    for kx in range(size // 4, size, size // 3):
        rope[rope_y : rope_y + rope_thick + 2, kx : kx + 2, :] = (255, 255, 255)
    sprites["-"] = rope

    # 6. Gold ('G') - Gleaming golden treasure box with gem emblem
    gold = np.zeros((size, size, 3), dtype=np.uint8)
    box_pad = size // 5
    gold[box_pad : size - box_pad, box_pad : size - box_pad, :] = (255, 205, 10)
    # Highlight border
    gold[box_pad : box_pad + 2, box_pad : size - box_pad, :] = (255, 255, 190)
    gold[box_pad : size - box_pad, box_pad : box_pad + 2, :] = (255, 255, 190)
    # Dark bottom/right edge
    gold[size - box_pad - 2 : size - box_pad, box_pad : size - box_pad, :] = (160, 115, 0)
    gold[box_pad : size - box_pad, size - box_pad - 2 : size - box_pad, :] = (160, 115, 0)
    # Inner dark emblem
    mid = size // 2
    gold[mid - 2 : mid + 2, mid - 2 : mid + 2, :] = (255, 255, 255) # Bright glint
    sprites["G"] = gold

    # 7. Enemy ('E') - Retro red robot guard with white head and visor
    enemy = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    # Head
    enemy[3:7, cx - 3 : cx + 4, :] = (255, 220, 190)
    enemy[4:6, cx - 2 : cx + 3, :] = (255, 30, 30) # Visor/eyes
    # Torso (Bright Red)
    enemy[7:14, cx - 4 : cx + 5, :] = (235, 35, 35)
    enemy[9:12, cx - 2 : cx + 3, :] = (255, 255, 255) # Belt/badge
    # Arms
    enemy[8:12, cx - 6 : cx - 4, :] = (255, 220, 190)
    enemy[8:12, cx + 5 : cx + 7, :] = (255, 220, 190)
    # Legs (Running stance)
    enemy[14:20, cx - 4 : cx - 1, :] = (180, 20, 20)
    enemy[14:20, cx + 1 : cx + 4, :] = (180, 20, 20)
    enemy[19:22, cx - 6 : cx - 2, :] = (255, 255, 255) # Boots
    enemy[19:22, cx + 2 : cx + 6, :] = (255, 255, 255)
    sprites["E"] = enemy

    # 8. Player ('M') - Retro green runner hero with white head and cyan boots
    player = np.zeros((size, size, 3), dtype=np.uint8)
    # Head
    player[3:7, cx - 3 : cx + 4, :] = (255, 230, 200)
    player[4:6, cx - 1 : cx + 3, :] = (30, 140, 255) # Blue visor/cap
    # Torso (Vibrant Lime Runner Suit)
    player[7:14, cx - 4 : cx + 5, :] = (40, 230, 90)
    player[9:11, cx - 2 : cx + 3, :] = (255, 255, 255) # White chest stripe
    # Arms
    player[8:12, cx - 6 : cx - 4, :] = (255, 230, 200)
    player[8:12, cx + 5 : cx + 7, :] = (255, 230, 200)
    # Running legs
    player[14:19, cx - 4 : cx - 1, :] = (25, 170, 60)
    player[14:19, cx + 1 : cx + 4, :] = (25, 170, 60)
    # White/cyan running sneakers
    player[19:22, cx - 5 : cx - 1, :] = (220, 255, 255)
    player[19:22, cx + 1 : cx + 5, :] = (220, 255, 255)
    sprites["M"] = player

    return sprites


SPRITES = create_tile_sprites(TILE_SIZE)


def render_authentic_level(
    grid: List[str],
    reachable_cells: Optional[set] = None,
    unreachable_gold: Optional[list] = None,
    path: Optional[list] = None,
    current_step: Optional[int] = None,
) -> Image.Image:
    """Render 22x32 character level using arcade pixel-art sprites into a PIL Image."""
    R = len(grid)
    C = len(grid[0])
    img_h = R * TILE_SIZE
    img_w = C * TILE_SIZE

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for r in range(R):
        for c in range(C):
            tile = grid[r][c]
            sprite = SPRITES.get(tile, SPRITES["."])
            y1, y2 = r * TILE_SIZE, (r + 1) * TILE_SIZE
            x1, x2 = c * TILE_SIZE, (c + 1) * TILE_SIZE
            canvas[y1:y2, x1:x2] = sprite

    pil_img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(pil_img, mode="RGBA")

    # Reachable cells overlay (emerald / cyan tint)
    if reachable_cells:
        for r, c in reachable_cells:
            # Skip tinting solid ground so brick textures remain vibrant
            if grid[r][c] in {".", "-", "#", "G", "M", "E"}:
                y1, y2 = r * TILE_SIZE, (r + 1) * TILE_SIZE
                x1, x2 = c * TILE_SIZE, (c + 1) * TILE_SIZE
                draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=(0, 255, 200, 35))

    # Unreachable gold highlight (pulsing red target ring)
    if unreachable_gold:
        for gr, gc in unreachable_gold:
            cx = gc * TILE_SIZE + TILE_SIZE // 2
            cy = gr * TILE_SIZE + TILE_SIZE // 2
            rad = TILE_SIZE // 2 - 1
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 25, 40, 240), width=3)

    # Solver A* tour path
    if path and len(path) > 1:
        path_to_draw = path[: (current_step + 1)] if current_step is not None else path
        if len(path_to_draw) > 1:
            points = [
                (c * TILE_SIZE + TILE_SIZE // 2, r * TILE_SIZE + TILE_SIZE // 2)
                for r, c in path_to_draw
            ]
            draw.line(points, fill=(50, 255, 120, 210), width=2)

        # Current player position indicator
        if current_step is not None and current_step < len(path):
            curr_r, curr_c = path[current_step]
            cx = curr_c * TILE_SIZE + TILE_SIZE // 2
            cy = curr_r * TILE_SIZE + TILE_SIZE // 2
            draw.ellipse([cx - 8, cy - 8, cx + 8, cy + 8], fill=(0, 255, 130, 240), outline=(255, 255, 255, 255), width=2)

    return pil_img
