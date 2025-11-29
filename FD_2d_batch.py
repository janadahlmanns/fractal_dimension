import os
import sys
import math
from pathlib import Path
import cv2

import numpy as np
from PIL import Image, ImageFilter
import pandas as pd


# ------------------------------
# Fractal dimension (2D box-counting on edges)
# ------------------------------

def fractal_dimension_boxcount(binary_img: np.ndarray, min_box_size: int = 2) -> float:
    """
    Box-counting fractal dimension for a 2D binary edge image.

    binary_img: 2D numpy array of booleans or {0,1}
    returns: estimated fractal dimension (float in [0, 2])
    """
    if binary_img.ndim != 2:
        raise ValueError("Input to fractal_dimension_boxcount must be a 2D array.")

    # Ensure boolean
    img = binary_img.astype(bool)
    h, w = img.shape

    # If there are no edges at all → dimension 0
    if not img.any():
        return 0.0

    M = min(h, w)

    # Choose box sizes as powers of two
    max_box_size = M // 2
    if max_box_size < min_box_size:
        # Image too small for multi-scale estimate
        return 0.0

    box_sizes = []
    s = min_box_size
    while s <= max_box_size:
        box_sizes.append(s)
        s *= 2

    if len(box_sizes) < 2:
        # Not enough scales to do a regression
        return 0.0

    Ns = []

    for s in box_sizes:
        n_rows = math.ceil(h / s)
        n_cols = math.ceil(w / s)

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

                # Count box if it contains at least one edge pixel
                if block.any():
                    N_s += 1

        # If for some very large box size we get zero boxes, skip it
        if N_s > 0:
            Ns.append(N_s)
        else:
            # No boxes occupied at this scale → ignore this scale
            pass

    if len(Ns) < 2:
        return 0.0

    box_sizes = np.array(box_sizes[:len(Ns)], dtype=np.float64)
    Ns = np.array(Ns, dtype=np.float64)

    log_s = np.log(1.0 / box_sizes)
    log_N = np.log(Ns)

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
        return pd.DataFrame(columns=["filename", "fd_edge"])


def save_results(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)


def process_folder(folder: Path):
    folder = folder.resolve()
    print(f"Processing folder: {folder}")

    csv_path = folder / "fd_results_edges.csv"
    edges_dir = folder / "edges"
    edges_dir.mkdir(exist_ok=True)

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
            # 1) Load and convert to grayscale
            with Image.open(img_path) as im:
                im_gray = im.convert("L")

            # 2) Proper edge detection using OpenCV Canny
            arr = np.array(im_gray)

            # Smooth → reduces tiny noise edges, HUGE improvement
            blur = cv2.GaussianBlur(arr, (5, 5), 1.6)

            # Canny edges (these numbers are a good default)
            edges = cv2.Canny(blur, threshold1=30, threshold2=110)

            # 3) Binarize
            binary = edges > 0

            # 4) Save edge image as PNG
            edges_uint8 = (binary.astype(np.uint8) * 255)
            edges_img = Image.fromarray(edges_uint8, mode="L")
            edge_path = edges_dir / (img_path.stem + "_edges.png")
            edges_img.save(edge_path)

            # 5) Compute FD
            fd = fractal_dimension_boxcount(binary)

            print(f"  -> FD_edge = {fd:.4f}")

            # 6) Append to results
            results_df = pd.concat(
                [results_df,
                 pd.DataFrame([{"filename": rel_name, "fd_edge": fd}])],
                ignore_index=True
            )
            already_done.add(rel_name)
            save_results(results_df, csv_path)

        except Exception as e:
            print(f"  !! Error with {rel_name}: {e}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compute_fd_edges_batch.py /path/to/image_folder")
        sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    process_folder(folder)
