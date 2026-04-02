#!/usr/bin/env python3
"""
Normalize sprite animation frames using ML background removal (rembg).
1. Remove checkerboard background -> true RGBA transparency
2. Normalize all frames of an entity to same scale & bottom-center alignment
"""

import os
import glob
import sys
from PIL import Image
from rembg import remove, new_session
import numpy as np

OUTPUT_SIZE = (512, 512)
CONTENT_RATIO = 0.80  # Character fills 80% of canvas height max
BOTTOM_PAD = 0.05     # 5% padding from bottom


def get_content_bbox(rgba_img):
    """Get bounding box of non-transparent content."""
    alpha = np.array(rgba_img.split()[3])
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax + 1), int(rmax + 1))


def normalize_entity_frames(entity_dir, session):
    """Normalize all frames in an entity directory."""
    frames_dir = os.path.join(entity_dir, "frames")
    if not os.path.isdir(frames_dir):
        return

    png_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not png_files:
        return

    entity_name = entity_dir.replace("png/", "")
    print(f"\n=== {entity_name} ({len(png_files)} frames) ===")

    # Step 1: Remove backgrounds and collect bounding boxes
    rgba_images = {}
    bboxes = {}

    for i, f in enumerate(png_files):
        fname = os.path.basename(f)
        img = Image.open(f)
        rgba = remove(img, session=session)
        bbox = get_content_bbox(rgba)
        rgba_images[fname] = rgba
        bboxes[fname] = bbox

        if bbox:
            cw = bbox[2] - bbox[0]
            ch = bbox[3] - bbox[1]
            print(f"  [{i+1}/{len(png_files)}] {fname}: {cw}x{ch}", flush=True)
        else:
            print(f"  [{i+1}/{len(png_files)}] {fname}: NO CONTENT", flush=True)

    # Step 2: Find max content dimensions across ALL frames of this entity
    valid_bboxes = [b for b in bboxes.values() if b is not None]
    if not valid_bboxes:
        print("  SKIP: no content")
        return

    max_cw = max(b[2] - b[0] for b in valid_bboxes)
    max_ch = max(b[3] - b[1] for b in valid_bboxes)
    print(f"  Max content: {max_cw}x{max_ch}")

    # Step 3: Scale factor
    target_h = int(OUTPUT_SIZE[1] * CONTENT_RATIO)
    target_w = int(OUTPUT_SIZE[0] * CONTENT_RATIO)
    scale = min(target_w / max_cw, target_h / max_ch)
    print(f"  Scale: {scale:.3f}")

    # Step 4: Process each frame
    for fname, rgba in rgba_images.items():
        bbox = bboxes[fname]
        if bbox is None:
            blank = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
            blank.save(os.path.join(frames_dir, fname))
            continue

        # Crop to content
        content = rgba.crop(bbox)
        cw, ch = content.size

        # Scale uniformly
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        content = content.resize((new_w, new_h), Image.LANCZOS)

        # Place on canvas: bottom-center aligned
        canvas = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        x = (OUTPUT_SIZE[0] - new_w) // 2
        y = OUTPUT_SIZE[1] - int(OUTPUT_SIZE[1] * BOTTOM_PAD) - new_h
        canvas.paste(content, (x, y), content)
        canvas.save(os.path.join(frames_dir, fname))

    print(f"  -> Done! {len(png_files)} frames normalized")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base)

    # Create rembg session once (loads model once)
    print("Loading rembg model...")
    session = new_session("u2net")
    print("Model loaded.\n")

    entity_dirs = []
    for root, dirs, files in os.walk("png"):
        if "frames" in dirs:
            entity_dirs.append(root)

    entity_dirs.sort()
    print(f"Found {len(entity_dirs)} entities to normalize")

    total_frames = sum(
        len(glob.glob(os.path.join(d, "frames", "*.png")))
        for d in entity_dirs
    )
    print(f"Total frames: {total_frames}")

    for ed in entity_dirs:
        normalize_entity_frames(ed, session)

    print("\nAll done!")


if __name__ == "__main__":
    main()
