#!/usr/bin/env python3
"""Process a sprite sheet: split into frames, remove background, normalize to 512x512."""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

OUTPUT_SIZE = (512, 512)
CONTENT_RATIO = 0.85
BOTTOM_PAD = 0.03


def detect_layout(img, num_frames):
    """Detect sprite sheet layout by checking for white bands between rows."""
    w, h = img.size
    arr = np.array(img)

    def has_horizontal_white_band(y_center, search_range=30):
        """Check if there's a white band near the given y position."""
        # Search a range around the expected split point
        for y in range(max(0, y_center - search_range), min(h, y_center + search_range)):
            row_white = np.mean(arr[y, :, :] > 230)
            if row_white > 0.95:
                return True
        return False

    if num_frames == 4:
        if has_horizontal_white_band(h // 2):
            return (2, 2)
        return (1, 4)
    elif num_frames == 6:
        if has_horizontal_white_band(h // 2):
            return (2, 3)
        return (1, 6)
    elif num_frames == 8:
        if has_horizontal_white_band(h // 2):
            return (2, 4)
        return (1, 8)
    elif num_frames == 3:
        return (1, 3)
    return (1, num_frames)


def split_sheet(img, num_frames):
    """Split sprite sheet into individual frames."""
    rows, cols = detect_layout(img, num_frames)
    w, h = img.size
    frame_w = w // cols
    frame_h = h // rows

    frames = []
    for r in range(rows):
        for c in range(cols):
            if len(frames) >= num_frames:
                break
            left = c * frame_w
            top = r * frame_h
            frame = img.crop((left, top, left + frame_w, top + frame_h))
            frames.append(frame)

    return frames


def remove_background(img):
    """Remove white/near-white background."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    arr = np.array(img)
    r, g, b = arr[:,:,0].astype(int), arr[:,:,1].astype(int), arr[:,:,2].astype(int)

    white = (r > 235) & (g > 235) & (b > 235)
    gray = (r > 215) & (g > 215) & (b > 215) & (np.abs(r-g) < 15) & (np.abs(g-b) < 15)
    bg = white | gray

    arr[bg, 3] = 0
    return Image.fromarray(arr)


def get_content_bbox(img):
    arr = np.array(img)
    alpha = arr[:,:,3]
    rows = np.any(alpha > 20, axis=1)
    cols = np.any(alpha > 20, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (cmin, rmin, cmax + 1, rmax + 1)


def normalize_and_save(frames, output_paths):
    """Normalize frames to consistent size on OUTPUT_SIZE canvas, save."""
    rgba_frames = [remove_background(f) for f in frames]
    bboxes = [get_content_bbox(f) for f in rgba_frames]

    valid = [(i, b) for i, b in enumerate(bboxes) if b is not None]
    if not valid:
        print("  WARNING: No content detected in any frame!")
        for i, path in enumerate(output_paths):
            canvas = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
            canvas.save(path)
        return

    max_cw = max(bboxes[i][2] - bboxes[i][0] for i, _ in valid)
    max_ch = max(bboxes[i][3] - bboxes[i][1] for i, _ in valid)

    target_w = int(OUTPUT_SIZE[0] * CONTENT_RATIO)
    target_h = int(OUTPUT_SIZE[1] * CONTENT_RATIO)
    scale = min(target_w / max_cw, target_h / max_ch)

    for i, (rgba, bbox, path) in enumerate(zip(rgba_frames, bboxes, output_paths)):
        canvas = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        if bbox is not None:
            content = rgba.crop(bbox)
            cw, ch = content.size
            new_w, new_h = int(cw * scale), int(ch * scale)
            if new_w > 0 and new_h > 0:
                content = content.resize((new_w, new_h), Image.LANCZOS)
                x = (OUTPUT_SIZE[0] - new_w) // 2
                y = OUTPUT_SIZE[1] - int(OUTPUT_SIZE[1] * BOTTOM_PAD) - new_h
                canvas.paste(content, (x, y), content)
        canvas.save(path)
        print(f"  Saved {path.name}")


def process_sprite_sheet(sheet_path, num_frames, output_dir, action_name):
    """Full pipeline: split -> remove bg -> normalize -> save."""
    img = Image.open(sheet_path)
    print(f"  Sheet: {img.size}, layout: {detect_layout(img, num_frames)}")

    frames = split_sheet(img, num_frames)
    output_paths = [
        Path(output_dir) / f"{action_name}_{str(i+1).zfill(2)}.png"
        for i in range(num_frames)
    ]

    normalize_and_save(frames, output_paths)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: process_sheet.py <sheet_path> <num_frames> <output_dir> <action_name>")
        sys.exit(1)

    sheet_path = sys.argv[1]
    num_frames = int(sys.argv[2])
    output_dir = sys.argv[3]
    action_name = sys.argv[4]

    process_sprite_sheet(sheet_path, num_frames, output_dir, action_name)
