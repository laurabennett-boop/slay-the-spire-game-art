#!/usr/bin/env python3
"""
Generate sprite sheets with Gemini, then split into individual frames.
Sprite sheet approach ensures consistent character size/position across frames.
"""

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_DIR = Path("/Users/visionclaw/game-art-preview")
OUTPUT_SIZE = (512, 512)
CONTENT_RATIO = 0.85  # character fills 85% of frame
BOTTOM_PAD = 0.03     # 3% bottom padding


def load_tasks():
    with open(PROJECT_DIR / "generation-tasks.json") as f:
        return json.load(f)


def split_sprite_sheet(img_path, num_frames):
    """Split a horizontal sprite sheet into equal-width frames."""
    img = Image.open(img_path)
    w, h = img.size
    frame_w = w // num_frames
    frames = []
    for i in range(num_frames):
        left = i * frame_w
        right = (i + 1) * frame_w
        frame = img.crop((left, 0, right, h))
        frames.append(frame)
    return frames


def remove_background(img):
    """Remove white/near-white background, return RGBA image."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    arr = np.array(img)

    # White background: R > 240 AND G > 240 AND B > 240
    r, g, b, a = arr[:,:,0], arr[:,:,1], arr[:,:,2], arr[:,:,3]
    white_mask = (r > 235) & (g > 235) & (b > 235)

    # Also catch light gray near-white
    gray_mask = (r > 220) & (g > 220) & (b > 220) & (np.abs(r.astype(int) - g.astype(int)) < 15) & (np.abs(g.astype(int) - b.astype(int)) < 15)

    bg_mask = white_mask | gray_mask
    arr[bg_mask, 3] = 0  # Set alpha to 0 for background

    return Image.fromarray(arr)


def get_content_bbox(img):
    """Get bounding box of non-transparent content."""
    arr = np.array(img)
    alpha = arr[:,:,3]
    rows = np.any(alpha > 20, axis=1)
    cols = np.any(alpha > 20, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax + 1, rmax + 1)


def normalize_frames_on_canvas(frames):
    """
    Normalize all frames to OUTPUT_SIZE canvas with consistent sizing.
    Uses max content dimensions across ALL frames for uniform scale.
    Bottom-center alignment for stable foot position.
    """
    # Remove backgrounds
    rgba_frames = [remove_background(f) for f in frames]

    # Get bounding boxes
    bboxes = [get_content_bbox(f) for f in rgba_frames]
    valid = [(f, b) for f, b in zip(rgba_frames, bboxes) if b is not None]

    if not valid:
        return rgba_frames  # fallback

    # Find max content dimensions
    max_cw = max(b[2] - b[0] for _, b in valid)
    max_ch = max(b[3] - b[1] for _, b in valid)

    # Calculate uniform scale
    target_w = int(OUTPUT_SIZE[0] * CONTENT_RATIO)
    target_h = int(OUTPUT_SIZE[1] * CONTENT_RATIO)
    scale = min(target_w / max_cw, target_h / max_ch)

    result = []
    for rgba, bbox in zip(rgba_frames, bboxes):
        canvas = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        if bbox is None:
            result.append(canvas)
            continue

        # Crop content
        content = rgba.crop(bbox)
        cw, ch = content.size
        new_w = int(cw * scale)
        new_h = int(ch * scale)
        content = content.resize((new_w, new_h), Image.LANCZOS)

        # Bottom-center align
        x = (OUTPUT_SIZE[0] - new_w) // 2
        y = OUTPUT_SIZE[1] - int(OUTPUT_SIZE[1] * BOTTOM_PAD) - new_h
        canvas.paste(content, (x, y), content)
        result.append(canvas)

    return result


def build_sprite_sheet_prompt(style_prefix, entity_visual, action_name, num_frames, descriptions):
    """Build a prompt for generating a sprite sheet."""
    frame_desc = "\n".join(f"Panel {i+1}: {d}" for i, d in enumerate(descriptions))

    return f"""Sprite sheet: {num_frames}-frame {action_name} animation sequence.

Character: {entity_visual}

Show exactly {num_frames} panels arranged horizontally in a single row, each panel the same size. The character appears at the EXACT SAME SIZE, SAME POSITION, and SAME PROPORTIONS in every panel. Only the pose changes between panels to create animation frames:
{frame_desc}

Style: {style_prefix}
Solid white background. Full body visible in each panel, character centered in each panel. Professional game sprite asset sheet, consistent character design across all panels."""


def main():
    tasks = load_tasks()
    style_prefix = tasks["style_prefix"]
    entities = tasks["entities"]

    # Check which entities/actions to process
    skip_existing = "--skip-existing" in sys.argv
    only_entity = None
    only_action = None
    for arg in sys.argv[1:]:
        if not arg.startswith("--"):
            parts = arg.split("/")
            only_entity = parts[0]
            if len(parts) > 1:
                only_action = parts[1]

    total_sheets = sum(len(e["actions"]) for e in entities.values())
    current = 0

    for entity_key, entity in entities.items():
        if only_entity and entity_key != only_entity:
            continue

        entity_name = entity["name"]
        base_path = PROJECT_DIR / entity["base_path"]
        base_path.mkdir(parents=True, exist_ok=True)

        for action_name, action in entity["actions"].items():
            if only_action and action_name != only_action:
                continue

            current += 1
            num_frames = action["frames"]
            descriptions = action["descriptions"]

            # Check if frames already exist
            first_frame = base_path / f"{action_name}_01.png"
            if skip_existing and first_frame.exists():
                print(f"[{current}/{total_sheets}] SKIP {entity_name} / {action_name} (exists)")
                continue

            print(f"[{current}/{total_sheets}] Generating {entity_name} / {action_name} ({num_frames} frames)...")

            # Build prompt
            prompt = build_sprite_sheet_prompt(
                style_prefix, entity["visual"], action_name, num_frames, descriptions
            )

            # Call Gemini via MCP tool (we'll use subprocess to call claude)
            # For now, just print the prompt - actual generation will be orchestrated by the agent
            print(f"  Prompt ready ({len(prompt)} chars)")
            print(f"  Output: {base_path}/{action_name}_*.png")

    print(f"\nDone! Processed {current} sprite sheets.")


if __name__ == "__main__":
    main()
