import os
import sys
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pandas as pd


# ------------------------------
# Fractal dimension (DBC-style)
# ------------------------------

def fractal_dimension_dbc(gray_img: np.ndarray, min_box_size: int = 2) -> float:
    """
    Differential box-counting fractal dimension for a grayscale image.

    gray_img: 2D numpy array, values 0-255
    returns: estimated fractal dimension (float)
    """
    if gray_img.ndim != 2:
        raise ValueError("Input to fractal_dimension_dbc must be a 2D array (grayscale).")

    h, w = gray_img.shape

    # Normalize intensities to [0, 255]
    img = gray_img.astype(np.float64)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) * (255.0 / (img_max - img_min))

    M = min(h, w)       # spatial size (for aspect ratio)
    G = 256.0           # gray levels

    # Choose box sizes as powers of two
    max_box_size = M // 2
    if max_box_size < min_box_size:
        raise ValueError("Image too small for chosen min_box_size.")

    box_sizes = []
    s = min_box_size
    while s <= max_box_size:
        box_sizes.append(s)
        s *= 2

    if len(box_sizes) < 2:
        return 2.0  # fallback

    Ns = []

    for s in box_sizes:
        # Number of boxes along each spatial dimension
        n_rows = math.ceil(h / s)
        n_cols = math.ceil(w / s)

        # *** KEY FIX ***
        # Make intensity step proportional to spatial box size,
        # so boxes are roughly cubic in (x, y, gray).
        dz = math.ceil(G * s / M)

        N_s = 0
        for i in range(n_rows):
            for j in range(n_cols):
                r_start = i * s
                c_start = j * s
                r_end = min((i + 1) * s, h)
                c_end = min((j + 1) * s, w)

                block = img[r_start:r_end, c_start:c_end]
                if block.size == 0:
                    continue

                g_min = block.min()
                g_max = block.max()

                k_min = math.floor(g_min / dz)
                k_max = math.floor(g_max / dz)
                n_block_boxes = (k_max - k_min) + 1
                N_s += n_block_boxes

        Ns.append(N_s)

    # Fit line to log-log plot: log N(s) vs log(1/s)
    log_s = np.log(1 / np.array(box_sizes, dtype=np.float64))
    log_N = np.log(np.array(Ns, dtype=np.float64))

    coeffs = np.polyfit(log_s, log_N, 1)
    D = coeffs[0]
    return float(D)



# ------------------------------
# Batch processing
# ------------------------------

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_results(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return pd.DataFrame(columns=["filename", "fd"])


def save_results(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)


def process_folder(folder: Path):
    folder = folder.resolve()
    print(f"Processing folder: {folder}")

    csv_path = folder / "fd_3d_results.csv"
    gray_dir = folder / "gray"
    gray_dir.mkdir(exist_ok=True)

    results_df = load_results(csv_path)
    already_done = set(results_df["filename"].tolist())

    image_files = [p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in VALID_EXTS]

    if not image_files:
        print("No image files found.")
        return

    for img_path in sorted(image_files):
        rel_name = img_path.name

        if rel_name in already_done:
            print(f"Skipping (already done): {rel_name}")
            continue

        print(f"Processing: {rel_name}")

        try:
            # Load and convert to grayscale
            with Image.open(img_path) as im:
                im_gray = im.convert("L")  # grayscale

            # Save grayscale image
            gray_path = gray_dir / (img_path.stem + "_gray.png")
            im_gray.save(gray_path)

            # Compute FD
            arr = np.array(im_gray)
            fd = fractal_dimension_dbc(arr)

            print(f"  -> FD = {fd:.4f}")

            # Append to results
            results_df = pd.concat(
                [results_df,
                 pd.DataFrame([{"filename": rel_name, "fd": fd}])],
                ignore_index=True
            )
            already_done.add(rel_name)
            save_results(results_df, csv_path)

        except Exception as e:
            print(f"  !! Error with {rel_name}: {e}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_fd_batch.py /path/to/image_folder")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    process_folder(folder)
