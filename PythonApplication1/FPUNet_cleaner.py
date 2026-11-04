"""
fpunet_cleaner.py

Runs a trained FPUNet (P2 Explore, Song et al., IROS 2025) on your occupancy grid
and uses its answer to decide which wall pixels are real walls and which are
furniture / clutter.

How it plugs into your grid
---------------------------
FPUNet was trained on 0.2 m cells in 128 x 128 pieces, with pixel values
255 = free, 0 = occupied, 100 = unknown. Your SLAM grid is finer (0.05 m), so:

  1. the grid is shrunk to 0.2 m cells (a cell is a wall if ANY fine cell in it is a wall),
  2. FPUNet cleans it piece by piece and the pieces are stitched together,
  3. every fine wall pixel is kept only if its 0.2 m cell is still a wall after FPUNet.

So FPUNet only decides what to ERASE, and the surviving walls keep their exact
5 cm geometry. FPUNet never draws new wall pixels here, because at 0.2 m it
cannot place them precisely.

Needs: torch, torchvision, and FPUNet.py copied from P2Explore/utils/FPUNet.py
into your project folder, plus a trained .pth file.
"""

import os
import sys
import importlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage

# FPUNet's pixel values
FREE, OCC, UNK = 255, 0, 100

TARGET_RES_M = 0.2   # the cell size FPUNet was trained on
PATCH = 128          # FPUNet input size (cells)
STRIDE = 64          # overlap between pieces, so seams are averaged out
BATCH = 16           # pieces per forward pass

_CMAP = ListedColormap(["#C8C8C8", "#FFFFFF", "#000000"])   # unknown, free, wall


# ============================================================
# LOAD THE MODEL
# ============================================================

def load_fpunet(weights_path, device=None):
    """Returns (net, device). Raises an error that says exactly what failed and which Python ran."""
    import torch

    # Look for FPUNet.py next to THIS file too, so it is found no matter which
    # folder the script is started from (IDEs often change the working directory).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    FPUNet = None
    errors = []
    for module_name in ("FPUNet", "utils.FPUNet"):
        try:
            FPUNet = importlib.import_module(module_name).FPUNet
            break
        except ImportError as e:
            errors.append(f"  import {module_name}: {e}")

    if FPUNet is None:
        raise ImportError(
            "Could not import FPUNet. Details:\n" + "\n".join(errors) + "\n"
            f"  python running this script: {sys.executable}\n"
            f"  looked in: {here} and {os.getcwd()}\n"
            "Copy FPUNet.py (P2Explore/utils/FPUNet.py) into one of those folders, and install "
            "torch and torchvision for THAT python."
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = FPUNet(in_channels=3, n_classes=3, feature_scale=2)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return net.to(device).eval(), device


# ============================================================
# GRID <-> FPUNET IMAGE
# ============================================================

def _to_fpunet_values(grid):
    """-1/0/1 grid -> 255/0/100 image (unknown/free/wall -> 100/255/0)."""
    img = np.full(grid.shape, UNK, np.uint8)
    img[grid == 0] = FREE
    img[grid == 1] = OCC
    return img


def _downsample(img, k):
    """k x k blocks -> one cell: free if >= 50% free, wall if ANY cell is a wall, else unknown."""
    h, w = img.shape
    H, W = -(-h // k), -(-w // k)
    pad = np.full((H * k, W * k), UNK, np.uint8)
    pad[:h, :w] = img
    blocks = pad.reshape(H, k, W, k).transpose(0, 2, 1, 3).reshape(H, W, k * k)
    small = np.full((H, W), UNK, np.uint8)
    small[(blocks == FREE).mean(-1) >= 0.5] = FREE
    small[(blocks == OCC).any(-1)] = OCC
    return small


def _encode(piece):
    """128 x 128 piece -> the 3 input layers FPUNet expects: free, occupied, unknown."""
    return np.stack([piece > 200, piece < 10, (piece > 90) & (piece < 110)]).astype(np.float32)


def _predict(small, net, device):
    """Runs FPUNet over overlapping pieces and stitches the answers."""
    import torch

    h, w = small.shape
    HH = PATCH if h <= PATCH else -(-(h - PATCH) // STRIDE) * STRIDE + PATCH
    WW = PATCH if w <= PATCH else -(-(w - PATCH) // STRIDE) * STRIDE + PATCH
    big = np.full((HH, WW), UNK, np.uint8)
    big[:h, :w] = small

    acc = np.zeros((3, HH, WW), np.float32)
    cnt = np.zeros((HH, WW), np.float32)
    coords = [(y, x)
              for y in range(0, HH - PATCH + 1, STRIDE)
              for x in range(0, WW - PATCH + 1, STRIDE)]

    with torch.no_grad():
        for i in range(0, len(coords), BATCH):
            chunk = coords[i:i + BATCH]
            batch = np.stack([_encode(big[y:y + PATCH, x:x + PATCH]) for y, x in chunk])
            out = net(torch.from_numpy(batch).to(device)).cpu().numpy()
            for (y, x), o in zip(chunk, out):
                acc[:, y:y + PATCH, x:x + PATCH] += o
                cnt[y:y + PATCH, x:x + PATCH] += 1

    idx = np.argmax(acc / cnt, axis=0)[:h, :w]
    pred = np.full((h, w), FREE, np.uint8)
    pred[idx == 1] = OCC
    pred[idx == 2] = UNK
    pred[small == UNK] = UNK    # keep the answer only inside the scanned area
    return pred


# ============================================================
# CLEAN
# ============================================================

def clean_with_fpunet(grid, resolution, net, device):
    """
    grid: int8 occupancy grid, -1 unknown / 0 free / 1 wall, row 0 = lowest Y
          (the format generate_raw_grid returns).
    Returns (cleaned_grid, info). cleaned_grid has the same shape and format;
    its wall pixels are always a subset of the input's wall pixels.
    """
    h, w = grid.shape
    k = max(1, int(round(TARGET_RES_M / resolution)))
    if abs(k * resolution - TARGET_RES_M) > 0.02:
        print(f"WARNING: {resolution} m/cell does not divide 0.2 m evenly "
              f"(using {k * resolution:.2f} m cells); FPUNet was trained on 0.2 m.")

    # FPUNet was trained on images (row 0 = top). Our grid has row 0 = lowest Y,
    # so flip it vertically first and flip the result back at the end.
    flipped = np.flipud(grid)
    small = _downsample(_to_fpunet_values(flipped), k)
    print(f"FPUNet: {resolution} m/cell -> {k * resolution:.2f} m cells "
          f"({small.shape[1]} x {small.shape[0]}), running model...")

    pred = _predict(small, net, device)

    keep_coarse = (pred == OCC)
    keep_fine = np.repeat(np.repeat(keep_coarse, k, axis=0), k, axis=1)[:h, :w]

    out = flipped.copy()
    erase = (flipped == 1) & ~keep_fine
    out[erase] = 0
    out = np.ascontiguousarray(np.flipud(out))

    n_in = int((grid == 1).sum())
    n_out = int((out == 1).sum())
    info = {
        "wall_pixels_in": n_in,
        "wall_pixels_out": n_out,
        "erased_fraction": (n_in - n_out) / n_in if n_in else 0.0,
        "coarse_shape": small.shape,
    }
    print(f"FPUNet erased {n_in - n_out} of {n_in} wall pixels ({info['erased_fraction']:.1%}).")
    if info["erased_fraction"] > 0.5:
        print("WARNING: FPUNet erased more than half of the walls. Check the weights file, "
              "and that the map resolution and orientation are what you expect.")
    return out, info


# ============================================================
# PLOTS
# ============================================================

def plot_three_way(raw_grid, classic_grid, fpunet_grid, show=True,
                   save_path="cleaning_comparison.png"):
    """FIGURE 1: raw / classic cleaning / FPUNet cleaning side by side (shared zoom)."""
    panels = [
        (raw_grid, "Raw occupancy grid"),
        (classic_grid, "Cleaned: classic filters"),
        (fpunet_grid, "Cleaned: FPUNet"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharex=True, sharey=True)
    for ax, (g, title) in zip(axes, panels):
        ax.imshow(g, cmap=_CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        ax.set_title(f"{title}\n{int((g == 1).sum())} wall pixels")
        ax.set_xlabel("X (cells)")
    axes[0].set_ylabel("Y (cells)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def _draw_cad(ax, grid, lines):
    """CAD-style map: white = free, light gray = unexplored, black vector walls."""
    bg = np.full(grid.shape, 255, dtype=np.uint8)
    bg[grid == -1] = 200
    ax.imshow(bg, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")
    for l in lines:
        ax.plot([l["x1"], l["x2"]], [l["y1"], l["y2"]],
                color="black", linewidth=3.0, solid_capstyle="round")
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_overlay(ax, grid, lines):
    """Wall lines in orange over the cleaned grid (good for judging against the pixels)."""
    ax.imshow(grid, cmap=_CMAP, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
    for l in lines:
        ax.plot([l["x1"], l["x2"]], [l["y1"], l["y2"]], color="#FF5733", linewidth=2.5)
    ax.set_xlabel("X (cells)")


def plot_vector_comparison(grid_a, lines_a, title_a, grid_b, lines_b, title_b,
                           style="cad", show=True, save_path="vector_comparison.png"):
    """
    FIGURE 2: the final vector map of each approach side by side.
    style="cad"     -> black walls on white (like the single vector map you had before)
    style="overlay" -> orange walls drawn over the cleaned grid pixels
    """
    draw = _draw_cad if style == "cad" else _draw_overlay
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True, sharey=True)
    for ax, g, lines, title in ((axes[0], grid_a, lines_a, title_a),
                                (axes[1], grid_b, lines_b, title_b)):
        draw(ax, g, lines)
        total = sum(l["length_meters"] for l in lines)
        ax.set_title(f"{title}\n{len(lines)} segments, {total:.1f} m of wall",
                     fontsize=14, weight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


# ============================================================
# NUMBERS FOR COMPARING TWO WALL SETS
# ============================================================

def _rasterize(lines, shape):
    m = np.zeros(shape, np.uint8)
    for l in lines:
        cv2.line(m, (int(round(l["x1"])), int(round(l["y1"]))),
                 (int(round(l["x2"])), int(round(l["y2"]))), 255, 1)
    return m > 0


def line_agreement(lines_a, lines_b, resolution, tol_m=0.3):
    """Fraction of A's wall length that lies within tol_m of some wall in B."""
    if not lines_a or not lines_b:
        return 0.0
    allp = lines_a + lines_b
    h = int(max(max(l["y1"], l["y2"]) for l in allp)) + 3
    w = int(max(max(l["x1"], l["x2"]) for l in allp)) + 3
    ma, mb = _rasterize(lines_a, (h, w)), _rasterize(lines_b, (h, w))
    if not ma.any() or not mb.any():
        return 0.0
    dist_to_b = ndimage.distance_transform_edt(~mb)
    return float((dist_to_b[ma] <= tol_m / resolution).mean())


def compare_line_sets(lines_a, name_a, lines_b, name_b, resolution, tol_m=0.3):
    """Prints a small comparison table. Agreement is NOT accuracy: neither side is ground truth."""
    len_a = sum(l["length_meters"] for l in lines_a)
    len_b = sum(l["length_meters"] for l in lines_b)
    a_in_b = line_agreement(lines_a, lines_b, resolution, tol_m)
    b_in_a = line_agreement(lines_b, lines_a, resolution, tol_m)

    print()
    print("==============================================")
    print("COMPARISON OF THE TWO APPROACHES")
    print("==============================================")
    print(f"{'':28s}{name_a:>14s}{name_b:>14s}")
    print(f"{'wall segments':28s}{len(lines_a):>14d}{len(lines_b):>14d}")
    print(f"{'total wall length (m)':28s}{len_a:>14.1f}{len_b:>14.1f}")
    print(f"{f'{name_a} walls found in {name_b}':28s}{a_in_b:>14.1%}")
    print(f"{f'{name_b} walls found in {name_a}':28s}{b_in_a:>14.1%}")
    print(f"(a wall counts as 'found' if it lies within {tol_m} m of a wall from the other approach)")
    print("Where the two disagree is where to look at the plots: that is where one of them")
    print("kept furniture/noise, or removed a real wall.")