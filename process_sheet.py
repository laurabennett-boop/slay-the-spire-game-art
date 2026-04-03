#!/usr/bin/env python3
"""Process a sprite sheet: split into frames, remove background, normalize to 512x512."""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage

OUTPUT_SIZE = (512, 512)
CONTENT_RATIO = 0.85
BOTTOM_PAD = 0.03


def detect_layout(img, num_frames):
    """Detect sprite sheet layout by checking for white bands between rows."""
    w, h = img.size
    arr = np.array(img)

    def has_horizontal_white_band(y_center, search_range=30):
        """Check if there's a white band near the given y position."""
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
        # Check 2×3 (2 rows, 3 cols) first
        if has_horizontal_white_band(h // 2):
            return (2, 3)
        # Check 3×2 (3 rows, 2 cols) for portrait images
        if has_horizontal_white_band(h // 3):
            return (3, 2)
        return (1, 6)
    elif num_frames == 8:
        # Check 2×4 first
        if has_horizontal_white_band(h // 2):
            return (2, 4)
        # Check 4×2 for portrait images
        if has_horizontal_white_band(h // 4):
            return (4, 2)
        return (1, 8)
    elif num_frames == 3:
        return (1, 3)
    return (1, num_frames)


def find_best_gutter(arr, axis, expected_pos, search_range=40):
    """Find the actual white gutter position near an expected grid line.

    Scans for the line with the highest white pixel ratio near the expected
    position and returns it. Falls back to expected_pos if no clear gutter found.

    axis: 0 for horizontal gutters (rows), 1 for vertical gutters (columns)
    """
    h, w = arr.shape[:2]
    best_pos = expected_pos
    best_score = 0.0

    for offset in range(-search_range, search_range + 1):
        pos = expected_pos + offset
        if pos < 0:
            continue
        if axis == 0 and pos >= h:
            continue
        if axis == 1 and pos >= w:
            continue

        if axis == 0:  # horizontal line (row)
            line = arr[pos, :, :]
        else:  # vertical line (column)
            line = arr[:, pos, :]

        white_ratio = np.mean(line > 230)
        if white_ratio > best_score:
            best_score = white_ratio
            best_pos = pos

    return best_pos


def split_sheet(img, num_frames):
    """Split sprite sheet into individual frames using content-aware gutter detection."""
    rows, cols = detect_layout(img, num_frames)
    w, h = img.size
    arr = np.array(img)
    frame_w = w // cols
    frame_h = h // rows

    # Find actual gutter positions (where white lines are)
    col_gutters = [0]  # left edge
    for c in range(1, cols):
        expected_x = c * frame_w
        actual_x = find_best_gutter(arr, axis=1, expected_pos=expected_x, search_range=min(30, frame_w // 6))
        col_gutters.append(actual_x)
    col_gutters.append(w)  # right edge

    row_gutters = [0]  # top edge
    for r in range(1, rows):
        expected_y = r * frame_h
        actual_y = find_best_gutter(arr, axis=0, expected_pos=expected_y, search_range=min(30, frame_h // 6))
        row_gutters.append(actual_y)
    row_gutters.append(h)  # bottom edge

    frames = []
    for r in range(rows):
        for c in range(cols):
            if len(frames) >= num_frames:
                break
            left = col_gutters[c]
            right = col_gutters[c + 1]
            top = row_gutters[r]
            bottom = row_gutters[r + 1]

            # Only apply a tiny trim (2%) at internal gutters to avoid
            # picking up the very edge of adjacent cells
            tiny_margin_x = int((right - left) * 0.02)
            tiny_margin_y = int((bottom - top) * 0.02)

            trim_left = tiny_margin_x if c > 0 else 0
            trim_right = tiny_margin_x if c < cols - 1 else 0
            trim_top = tiny_margin_y if r > 0 else 0
            trim_bottom = tiny_margin_y if r < rows - 1 else 0

            frame = img.crop((
                left + trim_left,
                top + trim_top,
                right - trim_right,
                bottom - trim_bottom
            ))
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


def remove_disconnected_artifacts(img, min_ratio=0.02):
    """Remove small disconnected blobs that are separate from the main character.

    Uses gentle erosion (1 iteration) to break only the thinnest connections,
    then keeps components that are large enough relative to the main character.
    Recovers with proportional dilation to avoid shrinking the main body.
    """
    arr = np.array(img)
    alpha = arr[:, :, 3]
    mask = alpha > 30

    if not mask.any():
        return img

    # Use only 1 iteration of erosion - just enough to break pixel-thin connections
    # from adjacent cell bleed, without destroying thin character elements
    eroded = ndimage.binary_erosion(mask, iterations=1)
    if not eroded.any():
        # Erosion removed everything - character is all thin lines, skip artifact removal
        return img

    labeled, num_features = ndimage.label(eroded)
    if num_features <= 1:
        return img

    # Find the largest component
    component_sizes = ndimage.sum(eroded, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    max_size = component_sizes[largest_label - 1]

    # Keep the largest component and any component >= 2% of its size
    # Use 2 iterations of dilation (matching 1 erosion + 1 extra to recover edges)
    keep_mask = np.zeros_like(mask)

    for i, size in enumerate(component_sizes):
        label = i + 1
        if size >= max_size * min_ratio:
            comp_mask = labeled == label
            comp_recovered = ndimage.binary_dilation(comp_mask, iterations=2)
            keep_mask |= (mask & comp_recovered)

    # Zero out alpha for removed components
    arr[~keep_mask, 3] = 0
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
    rgba_frames = [remove_disconnected_artifacts(remove_background(f)) for f in frames]
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
