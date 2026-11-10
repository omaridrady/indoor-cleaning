"""
door_outline_prototype.py  -  PROTOTYPE, a separate step next to main.py.

Idea: instead of joining loose wall lines, outline the free space the robot saw,
then cut the outline wherever a door is found, so the border is connected by
construction and doors stay open.

  1. Load the same map main.py uses (BAG_PATH), clean it (FPUNet by default) and
     rotate it with the SAME angle main.py uses (estimate_dominant_angle on the
     classic-cleaned grid). Alignment code is not changed.
  2. Split free space into
       - solid free space: wide open areas the robot saw well
       - beam streaks:     thin rays of free space with unknown cells between them
     using a morphological opening (a streak is narrower than the kernel).
  3. Keep the largest solid region and trace its outline. Edges within a few
     degrees of 0/90 are squared and snapped onto the wall pixels; long walls that
     are really tilted keep their angle (the rotation itself is untouched).
  4. Label every point of the border by what lies just outside it:
       wall    -> wall pixels right there
       fan     -> beam streaks continue outward (the lidar saw through it)
       unseen  -> nothing known beyond
  5. Beam-fan doors: where the border leaves a wall, wanders mostly through beam
     fans and comes back to a wall within OPENING_MAX_M, the detour is replaced by
     a straight line across the gap. That line is a DOOR if it is door-sized,
     otherwise a wide OPENING. A plain fan stretch of door width between two walls
     is also a door.

Used two ways:
  - by main.py, as the third result (STEP 5b): outline(), export(),
    plot_vector_panels() and doors_closeup() are called on the grid main.py already made;
  - on its own (python door_outline_prototype.py): writes to
    outputs\\<map name>\\door_outline_prototype\\ with a comparison report.
Importing this file has no side effects: it does not import main.py at the top and
never changes the matplotlib backend.
"""
import json
import os

import cv2
import numpy as np
from scipy import ndimage

from FPUNet_cleaner import load_fpunet, clean_with_fpunet, _draw_cad
from occupancy_cleaner import clean_occupancy_grid
from room_polygon_builder import build_room_polygons
from ros_bag_reader import generate_raw_grid
from vector_export import merge_walls, _to_meters, _to_map_frame_matrix
from wall_line_processing import align_grid_to_walls, estimate_dominant_angle, process_wall_lines

# ============================================================
# SETTINGS (meters / degrees)
# ============================================================

CLEANER = "fpunet"            # standalone run only: "fpunet" or "classic" grid is outlined
                              # (main.py outlines FPUNet when it ran, otherwise classic)

STREAK_WIDTH_M = 0.25         # free strips narrower than this are beam streaks, not open space
HOLE_FILL_M2 = 0.5            # holes in the free space smaller than this are filled (noise, small furniture)
SIMPLIFY_M = 0.10             # outline simplification tolerance
AXIS_TOL_DEG = 4.0            # edges this close to 0/90 degrees are squared to 0/90
MIN_TILTED_M = 1.0            # off-axis edges at least this long keep their real angle (tilted walls)
MIN_EDGE_M = 0.25             # border edges shorter than this are removed (jagged steps)
SNAP_M = 0.25                 # a 0/90 edge may move this far to sit on wall pixels
WALL_NEAR_M = 0.20            # wall pixels this close outside the border -> "wall"
FAN_DEPTH_M = 0.80            # look this far outward for beam streaks -> "fan"
SMALL_GAP_M = 0.35            # non-wall stretches shorter than this, between walls, are wall breaks
DOOR_MIN_M, DOOR_MAX_M = 0.60, 1.60   # door width range for the beam-fan rule
OPENING_MAX_M = 5.0           # widest gap a beam-fan detour may be closed across
POCKET_MIN_FAN = 0.35         # a detour must be at least this fraction beam fan ...
POCKET_MAX_WALL = 0.50        # ... and at most this fraction wall, to be cut off
POCKET_STRONG_FAN = 0.60      # detours at least this fraction fan are clearly seen-through: the cut
                              # may cross free space (it has to, across an open doorway)

SHOW_PLOTS = False            # standalone run only: True = also open the figures in windows

KINDS = ("wall", "door", "opening", "unseen")
COLORS = {"wall": "black", "door": "#1E9E4A", "opening": "#E8871E", "unseen": "#9A9A9A"}


# ============================================================
# 1. LOAD, CLEAN, ALIGN (same calls as main.py)
# ============================================================

def load_and_align(map_yaml, weights_path):
    raw, _, _, min_x, min_y, res, _ = generate_raw_grid(map_yaml)
    classic = clean_occupancy_grid(raw, resolution=res)
    angle = estimate_dominant_angle(classic, res)          # exactly what main.py uses
    if CLEANER == "fpunet":
        net, device = load_fpunet(weights_path)
        cleaned, _ = clean_with_fpunet(raw, res, net, device)
    else:
        cleaned = classic
    aligned, rot, M = align_grid_to_walls(cleaned, res, angle_override=angle)
    raw_aligned, _, _ = align_grid_to_walls(raw, res, angle_override=angle)
    return dict(raw=raw_aligned, grid=aligned, res=res, angle=rot, M=M, origin=(min_x, min_y))


# ============================================================
# 2. SOLID FREE SPACE vs BEAM STREAKS
# ============================================================

def split_free_space(grid, res):
    free = (grid == 0).astype(np.uint8)
    k = max(3, int(round(STREAK_WIDTH_M / res)) | 1)
    solid = cv2.morphologyEx(free, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    lab, n = ndimage.label(solid, structure=np.ones((3, 3)))
    if n == 0:
        raise RuntimeError("No free space found in the map.")
    main_region = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    filled = ndimage.binary_fill_holes(main_region)
    holes, _ = ndimage.label(filled & ~main_region)
    area = np.bincount(holes.ravel()) * res * res
    small = area < HOLE_FILL_M2
    small[0] = False
    region = main_region | small[holes]
    streaks = (grid == 0) & ~region
    return region, streaks


# ============================================================
# 3. OUTLINE POLYGON (squared where the walls are square, tilted where they are tilted)
# ============================================================

def _edge_class(p, q):
    a = np.degrees(np.arctan2(q[1] - p[1], q[0] - p[0])) % 180.0
    if min(a, 180.0 - a) <= AXIS_TOL_DEG:
        return "h"
    if abs(a - 90.0) <= AXIS_TOL_DEG:
        return "v"
    return "d"


def _dedupe(P):
    keep = [P[i] for i in range(len(P)) if np.hypot(*(P[i] - P[i - 1])) > 0.5]
    return np.array(keep) if len(keep) >= 3 else P


SPIKE_MAX_M = 1.0             # spikes with a side shorter than this are outline artifacts
SPIKE_MAX_PX = [20.0]         # set from SPIKE_MAX_M and the map resolution in outline()


def _merge_straight(P):
    """Drops a vertex between two edges of the same 0/90 class, or two near-parallel tilted edges."""
    changed = True
    while changed and len(P) > 4:
        changed = False
        n = len(P)
        for i in range(n):
            a, b, c = P[i - 1], P[i], P[(i + 1) % n]
            ca, cb = _edge_class(a, b), _edge_class(b, c)
            same = ca == cb and ca in "hv"
            if ca == cb == "d":
                d1 = np.degrees(np.arctan2(*(b - a)[::-1])) % 180
                d2 = np.degrees(np.arctan2(*(c - b)[::-1])) % 180
                same = min(abs(d1 - d2), 180 - abs(d1 - d2)) < 3.0
            u, v = b - a, c - b
            nu, nv = np.hypot(*u), np.hypot(*v)
            # a short spike: the border turns almost straight back on itself (> ~160 deg)
            spike = nu > 0 and nv > 0 and np.dot(u, v) / (nu * nv) < -0.94 and min(nu, nv) < SPIKE_MAX_PX[0]
            if same or spike:
                P = np.delete(P, i, axis=0)
                changed = True
                break
    return P


def _square(P):
    """Makes 0/90 edges exactly horizontal / vertical (length-weighted per edge)."""
    P = P.copy()
    n = len(P)
    for cls, axis in (("h", 1), ("v", 0)):
        for i in range(n):
            j = (i + 1) % n
            if _edge_class(P[i], P[j]) == cls:
                m = (P[i][axis] + P[j][axis]) / 2.0
                P[i][axis] = m
                P[j][axis] = m
    return P


def _collapse_short(P, min_px):
    while len(P) > 4:
        n = len(P)
        L = np.array([np.hypot(*(P[(i + 1) % n] - P[i])) for i in range(n)])
        i = int(np.argmin(L))
        if L[i] >= min_px:
            break
        j = (i + 1) % n
        mid = (P[i] + P[j]) / 2.0
        P = np.delete(P, j, axis=0)
        P[i if j > i else i - 1] = mid
        P = _square(_merge_straight(_dedupe(P)))
    return P


def _snap(P, wall, res):
    """Moves each 0/90 edge onto the row / column of wall pixels it runs along, if there is one."""
    P = P.copy()
    h, w = wall.shape
    s = int(round(SNAP_M / res))
    n = len(P)
    for i in range(n):
        j = (i + 1) % n
        cls = _edge_class(P[i], P[j])
        if cls == "d":
            continue
        axis, along = (1, 0) if cls == "h" else (0, 1)
        lo, hi = sorted((P[i][along], P[j][along]))
        a, b = int(np.ceil(lo)) + 1, int(np.floor(hi)) - 1
        if b <= a:
            continue
        pos0 = P[i][axis]
        best, best_c = pos0, 0
        for d in range(-s, s + 1):
            r = int(round(pos0)) + d
            if not (0 <= r < (h if cls == "h" else w)):
                continue
            band = wall[max(r - 1, 0):r + 2, a:b].any(axis=0) if cls == "h" else wall[a:b, max(r - 1, 0):r + 2].any(axis=1)
            c = int(band.sum())
            if c > best_c or (c == best_c and abs(r - pos0) < abs(best - pos0)):
                best, best_c = float(r), c
        if best_c >= 0.3 * (b - a):
            P[i][axis] = best
            P[j][axis] = best
    return P


def outline_polygon(contour, wall, res):
    ap = cv2.approxPolyDP(contour.reshape(-1, 1, 2).astype(np.int32), SIMPLIFY_M / res, True).reshape(-1, 2).astype(float)
    P = []
    for i in range(len(ap)):
        p, q = ap[i], ap[(i + 1) % len(ap)]
        P.append(p)
        if _edge_class(p, q) == "d" and np.hypot(*(q - p)) < MIN_TILTED_M / res:
            P.append(np.array([q[0], p[1]]))      # short off-axis edge: make it a square step
    P = np.array(P)
    for _ in range(3):
        P = _square(_merge_straight(_dedupe(P)))
        P = _collapse_short(P, MIN_EDGE_M / res)
    P = _snap(P, wall, res)
    P = _collapse_short(_merge_straight(_dedupe(P)), MIN_EDGE_M / res)
    return P if len(P) >= 4 else None


# ============================================================
# 4. LABEL THE BORDER
# ============================================================

def _outward(p, q, region):
    d = (q - p) / max(np.hypot(*(q - p)), 1e-9)
    n = np.array([d[1], -d[0]])
    m = (p + q) / 2
    h, w = region.shape

    def inside(pt):
        x, y = int(round(pt[0])), int(round(pt[1]))
        return 0 <= x < w and 0 <= y < h and region[y, x]

    if inside(m + 3 * n) and not inside(m - 3 * n):
        return -n
    return n


def label_border(V, region, wall, streaks, res):
    """Samples the closed polygon every pixel: [point, direction, edge id, label, step length]."""
    h, w = region.shape
    near, deep = int(round(WALL_NEAR_M / res)), int(round(FAN_DEPTH_M / res))
    samples = []
    n = len(V)
    for k in range(n):
        p, q = V[k], V[(k + 1) % n]
        L = float(np.hypot(*(q - p)))
        if L < 1e-6:
            continue
        d = (q - p) / L
        out = _outward(p, q, region)
        lat = np.array([-out[1], out[0]])
        for j in range(int(np.ceil(L))):
            s = p + d * j
            hit = False
            for t in range(-1, near + 1):
                for l in (-1, 0, 1):
                    x, y = (s + out * t + lat * l).round().astype(int)
                    if 0 <= x < w and 0 <= y < h and wall[y, x]:
                        hit = True
                        break
                if hit:
                    break
            lab = "wall" if hit else "unseen"
            if not hit:
                cnt = 0
                for t in range(1, deep + 1):
                    for l in (-2, 0, 2):
                        x, y = (s + out * t + lat * l).round().astype(int)
                        if 0 <= x < w and 0 <= y < h and streaks[y, x]:
                            cnt += 1
                if cnt >= 3:
                    lab = "fan"
            samples.append([s, d, k, lab, min(1.0, L - j)])
    labs = [s[3] for s in samples]
    N = len(labs)
    order = ("wall", "fan", "unseen")      # fixed tie-break, so every run gives the same result
    for i in range(N):     # majority smoothing over 5 samples (cyclic)
        win = [labs[(i + o) % N] for o in range(-2, 3)]
        if win.count(labs[i]) < 3:
            samples[i][3] = max(order, key=lambda k: (win.count(k), -order.index(k)))
    return samples


def _runs(labels, lengths):
    """Cyclic runs: list of [label, start index, count, length]."""
    N = len(labels)
    start = next((i for i in range(N) if labels[i] != labels[i - 1]), None)
    if start is None:
        return [[labels[0], 0, N, float(sum(lengths))]]
    runs = []
    i = start
    for _ in range(N):
        if not runs or labels[i] != runs[-1][0]:
            runs.append([labels[i], i, 0, 0.0])
        runs[-1][2] += 1
        runs[-1][3] += lengths[i]
        i = (i + 1) % N
    return runs


def close_small_gaps(labels, lengths):
    for _ in range(3):
        runs = _runs(labels, lengths)
        if len(runs) < 3:
            break
        changed = False
        for i, r in enumerate(runs):
            if r[0] != "wall" and r[3] < SMALL_GAP_M and runs[i - 1][0] == "wall" and runs[(i + 1) % len(runs)][0] == "wall":
                for j in range(r[2]):
                    labels[(r[1] + j) % len(labels)] = "wall"
                changed = True
        if not changed:
            break
    return labels


# ============================================================
# 5. DOORS: CUT BEAM-FAN DETOURS, CLASSIFY GAPS
# ============================================================

def _gap_kind(width_m):
    if width_m < DOOR_MIN_M:
        return "wall"
    return "door" if width_m <= DOOR_MAX_M else "opening"


def _cuts_explored_space(pts, region_core):
    inside = total = 0
    h, w = region_core.shape
    for a, b in zip(pts[:-1], pts[1:]):
        L = int(np.hypot(*(b - a)))
        for t in np.linspace(0, 1, max(L, 2))[1:-1]:
            x, y = (a + (b - a) * t).round().astype(int)
            total += 1
            inside += 0 <= x < w and 0 <= y < h and region_core[y, x]
    return total > 0 and inside / total > 0.2


def _shortcut_line(A, B, region_core, res, strong_fan=False):
    """
    Line across the gap from A to B: straight first (doorways can sit in tilted walls),
    then the two one-corner versions. None if every option cuts through explored space,
    unless the detour is clearly seen-through (strong_fan), where the straight line is used.
    """
    options = [[A, B]]
    if min(abs(B[0] - A[0]), abs(B[1] - A[1])) * res > 0.30:
        options += [[A, np.array([B[0], A[1]]), B], [A, np.array([A[0], B[1]]), B]]
    for pts in options:
        if not _cuts_explored_space(pts, region_core):
            return pts
    return [A, B] if strong_fan else None


def find_doors(samples, region, res):
    """
    Returns the border as an ordered list of items:
      ("sample", index, kind)  or  ("cut", [points], kind)
    """
    labels = close_small_gaps([s[3] for s in samples], [s[4] * res for s in samples])
    lengths = [s[4] * res for s in samples]
    N = len(samples)
    runs = _runs(labels, lengths)
    core = cv2.erode(region.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    R = len(runs)
    removed = np.zeros(N, bool)
    cuts = {}        # start sample index of pocket -> (points, kind, end index)
    used = np.zeros(R, bool)

    def end_point(run):
        i = (run[1] + run[2] - 1) % N
        return samples[i][0] + samples[i][1] * samples[i][4]

    for i in range(R):
        if runs[i][0] != "wall" or used[i]:
            continue
        A = end_point(runs[i])
        best = None
        path = fan = wall_in = 0.0
        for step in range(1, R - 1):          # bounded by the 60 m path limit below
            j = (i + step) % R
            if used[j]:
                break
            r = runs[j]
            if r[0] == "wall" and step > 1:
                B = samples[r[1]][0]
                gap = float(np.hypot(*(B - A))) * res
                if gap <= OPENING_MAX_M and path >= 1.3 * gap and fan >= POCKET_MIN_FAN * path \
                        and wall_in <= POCKET_MAX_WALL * path:
                    pts = _shortcut_line(A, B, core, res, strong_fan=fan >= POCKET_STRONG_FAN * path)
                    if pts is not None:
                        best = (j, pts, gap)
            path += r[3]
            fan += r[3] if r[0] == "fan" else 0.0
            wall_in += r[3] if r[0] == "wall" else 0.0
            if path > 60.0:
                break
        if best is None:
            continue
        j, pts, gap = best
        start = (runs[i][1] + runs[i][2]) % N
        stop = runs[j][1]
        k = start
        while k != stop:
            removed[k] = True
            k = (k + 1) % N
        kind = _gap_kind(gap) if len(pts) == 2 else ("wall" if gap < DOOR_MIN_M else "opening")
        cuts[start] = (pts, kind)
        for m in range(1, (j - i) % R):
            used[(i + m) % R] = True

    # plain fan stretches (no detour) between two walls
    runs = _runs(labels, lengths)
    kinds = list(labels)
    for i, r in enumerate(runs):
        if r[0] == "fan":
            flanked = runs[i - 1][0] == "wall" and runs[(i + 1) % len(runs)][0] == "wall"
            if r[3] < DOOR_MIN_M:           # a sliver too narrow to pass through
                kind = "wall" if flanked else "unseen"
            else:
                kind = "door" if (flanked and r[3] <= DOOR_MAX_M) else "opening"
            for m in range(r[2]):
                kinds[(r[1] + m) % N] = kind

    items = []
    first = next((k for k in range(N) if not removed[k] and removed[k - 1]), 0) if removed.any() else 0
    k = first
    for _ in range(N):
        if k in cuts:
            items.append(("cut",) + cuts[k])
        if not removed[k]:
            items.append(("sample", k, kinds[k]))
        k = (k + 1) % N
    return items


def to_lines(items, samples, res):
    """Ordered border items -> straight segments, merging same-kind collinear neighbours."""
    segs = []   # [a, b, kind, edge key]
    for it in items:
        if it[0] == "cut":
            pts, kind = it[1], it[2]
            for a, b in zip(pts[:-1], pts[1:]):
                segs.append([np.array(a, float), np.array(b, float), kind, None])
        else:
            s = samples[it[1]]
            a, b = s[0], s[0] + s[1] * s[4]
            if segs and segs[-1][3] == (s[2], it[2]):
                segs[-1][1] = b
            else:
                segs.append([a.copy(), b, it[2], (s[2], it[2])])

    def direction(sg):
        v = sg[1] - sg[0]
        return np.degrees(np.arctan2(v[1], v[0])) % 180.0

    merged = []
    for sg in segs:
        if merged:
            pv = merged[-1]
            dd = abs(direction(pv) - direction(sg))
            if pv[2] == sg[2] and min(dd, 180 - dd) < 1.0 and np.hypot(*(pv[1] - sg[0])) < 1.5:
                pv[1] = sg[1]
                continue
        merged.append(sg)
    if len(merged) > 1:
        a, b = merged[-1], merged[0]
        dd = abs(direction(a) - direction(b))
        if a[2] == b[2] and min(dd, 180 - dd) < 1.0 and np.hypot(*(a[1] - b[0])) < 1.5:
            b[0] = a[0]
            merged.pop()
    lines = []
    for a, b, kind, _ in merged:
        L = float(np.hypot(*(b - a)))
        if L * res < 0.05:
            continue
        lines.append({"x1": float(a[0]), "y1": float(a[1]), "x2": float(b[0]), "y2": float(b[1]),
                      "length_pixels": L, "length_meters": L * res,
                      "angle_degrees": float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0),
                      "family": "horizontal" if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else "vertical", "kind": kind})
    return lines


def outline(grid, raw, res):
    region, streaks = split_free_space(grid, res)
    wall = grid == 1
    SPIKE_MAX_PX[0] = SPIKE_MAX_M / res
    contours, hier = cv2.findContours(region.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    lines, polys = [], []
    for c, hrow in zip(contours, hier[0]):
        if cv2.contourArea(c) * res * res < HOLE_FILL_M2:
            continue
        V = outline_polygon(c.reshape(-1, 2), wall, res)
        if V is None:
            continue
        polys.append(("outer" if hrow[3] < 0 else "hole", V))
        samples = label_border(V, region, wall, streaks, res)
        items = find_doors(samples, region, res)
        lines += to_lines(items, samples, res)
    return lines, polys, region, streaks


# ============================================================
# CURRENT VECTORISER (same calls as main.run_vector_pipeline), for comparison
# ============================================================

def current_vector(aligned, res, merge=True):
    wl, _ = process_wall_lines(aligned, resolution=res, save_result=False)
    snapped = build_room_polygons(aligned, wl, resolution=res, save_result=False)
    return merge_walls(snapped, res) if merge else snapped


# ============================================================
# NUMBERS
# ============================================================

def _rasterize(lines, shape):
    m = np.zeros(shape, np.uint8)
    for l in lines:
        cv2.line(m, (int(round(l["x1"])), int(round(l["y1"]))), (int(round(l["x2"])), int(round(l["y2"]))), 255, 1)
    return m > 0


def connectivity(lines, res, tol_m=0.15):
    """Number of separate pieces (lines touching within tol join) and loose wall ends."""
    if not lines:
        return 0, 0
    tol = tol_m / res
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
        return np.hypot(*(a + t * ab - p))

    P = [(np.array([l["x1"], l["y1"]]), np.array([l["x2"], l["y2"]])) for l in lines]
    loose = 0
    for i in range(n):
        for end in P[i]:
            touched = False
            for j in range(n):
                if i != j and seg_dist(end, *P[j]) <= tol:
                    touched = True
                    parent[find(i)] = find(j)
            if not touched and lines[i].get("kind", "wall") == "wall":
                loose += 1
    return len({find(i) for i in range(n)}), loose


def stretches(lines, kinds):
    """Consecutive segments of the same kind, in border order: [(kind, [segments]), ...]."""
    out = []        # [kind, segments, index of last segment]
    for i, l in enumerate(lines):
        if l["kind"] not in kinds:
            continue
        if out and out[-1][0] == l["kind"] and out[-1][2] == i - 1:
            out[-1][1].append(l)
            out[-1][2] = i
        else:
            out.append([l["kind"], [l], i])
    # the border is closed: a stretch running over the end joins the one at the start
    if len(out) > 1 and out[0][0] == out[-1][0] and out[0][1][0] is lines[0] and out[-1][2] == len(lines) - 1:
        out[0][1] = out[-1][1] + out[0][1]
        out.pop()
    return [(k, segs) for k, segs, _ in out]


def count_stretches(lines, kind):
    """Consecutive segments of the same kind count as one stretch (lines are in border order)."""
    return len(stretches(lines, (kind,)))


def evidence(lines, raw_wall, res, tol_m=0.15):
    if not lines:
        return 0.0, 0.0
    v = _rasterize(lines, raw_wall.shape)
    dw = ndimage.distance_transform_edt(~raw_wall)
    dv = ndimage.distance_transform_edt(~v)
    t = tol_m / res
    return float((dw[v] <= t).mean()), float((dv[raw_wall] <= t).mean())


# ============================================================
# EXPORT
# ============================================================

def export(lines, res, M, origin, rot, base, source_name="", cleaner=""):
    """Writes <base>.json / .geojson / .svg, same coordinate convention as vector_export.py."""
    segs, width, height, shift = _to_meters(lines, res)
    A = _to_map_frame_matrix(M, res, origin, shift)
    doc = {"format": "floorplan-outline-prototype-v0", "units": "meters",
           "axes": "x right, y up, origin at the bottom-left of the drawing",
           "source": source_name, "cleaner": cleaner,
           "width": round(width, 3), "height": round(height, 3), "rotation_applied_deg": round(float(rot), 3),
           "to_map_frame": [[round(float(v), 6) for v in row] for row in A]}
    for kind in KINDS:
        key = "unseen" if kind == "unseen" else kind + "s"
        doc[key] = [{"id": i, "x1": round(s[0], 3), "y1": round(s[1], 3), "x2": round(s[2], 3), "y2": round(s[3], 3),
                     "length": round(float(np.hypot(s[2] - s[0], s[3] - s[1])), 3)}
                    for i, s in enumerate(s for s, l in zip(segs, lines) if l["kind"] == kind)]
    with open(base + ".json", "w") as f:
        json.dump(doc, f, indent=2)
    feats = [{"type": "Feature", "properties": {"kind": l["kind"], "length_m": round(l["length_meters"], 3)},
              "geometry": {"type": "LineString", "coordinates": [[round(s[0], 3), round(s[1], 3)], [round(s[2], 3), round(s[3], 3)]]}}
             for s, l in zip(segs, lines)]
    with open(base + ".geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "properties": {"units": "meters"}, "features": feats}, f)
    pad = 0.5
    style = {"wall": 'stroke="black" stroke-width="0.12"', "door": 'stroke="#1E9E4A" stroke-width="0.12"',
             "opening": 'stroke="#E8871E" stroke-width="0.10"',
             "unseen": 'stroke="#9A9A9A" stroke-width="0.06" stroke-dasharray="0.15 0.12"'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-pad} {-pad} {width + 2 * pad:.3f} {height + 2 * pad:.3f}">',
             f'<rect x="{-pad}" y="{-pad}" width="{width + 2 * pad:.3f}" height="{height + 2 * pad:.3f}" fill="white"/>']
    for s, l in zip(segs, lines):
        parts.append(f'<line x1="{s[0]:.3f}" y1="{height - s[1]:.3f}" x2="{s[2]:.3f}" y2="{height - s[3]:.3f}" '
                     f'{style[l["kind"]]} stroke-linecap="round"/>')
    parts.append("</svg>")
    with open(base + ".svg", "w") as f:
        f.write("\n".join(parts))
    return doc


# ============================================================
# FIGURES
# ============================================================

def _draw(ax, lines, lw=2.6):
    for l in lines:
        st = dict(color=COLORS[l["kind"]], lw=lw if l["kind"] in ("wall", "door") else lw * 0.8, solid_capstyle="round")
        if l["kind"] == "unseen":
            st.update(ls=(0, (2, 2)), lw=lw * 0.55)
        ax.plot([l["x1"], l["x2"]], [l["y1"], l["y2"]], **st)


def plot_vector_panels(panels, show=True, save_path="vector_comparison.png"):
    """
    Side-by-side vector maps. panels: list of (grid, lines, title, style) where
    style "cad" = black walls (same look as FPUNet_cleaner.plot_vector_comparison)
    and "outline" = coloured wall / door / opening / unseen edges.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # size each panel to the map's shape, so tall maps are not squeezed into wide empty columns
    g0 = panels[0][0]
    ys, xs = np.nonzero(g0 >= 0)
    box = (xs.min() - 5, xs.max() + 5, ys.min() - 5, ys.max() + 5)
    aspect = (box[1] - box[0]) / max(box[3] - box[2], 1)
    height = 12.0
    panel_w = min(max(height * aspect, 4.0), 14.0)
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_w * len(panels) + 1, height + 1.2),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (grid, lines, title, style) in zip(axes, panels):
        if style == "cad":
            _draw_cad(ax, grid, lines)
            total = sum(l["length_meters"] for l in lines)
            ax.set_title(f"{title}\n{len(lines)} segments, {total:.1f} m of wall", fontsize=14, weight="bold")
        else:
            bg = np.full(grid.shape, 255, dtype=np.uint8)
            bg[grid == -1] = 200
            ax.imshow(bg, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")
            _draw(ax, lines, lw=3.0)
            ax.set_aspect("equal"); ax.axis("off")
            nd, no = count_stretches(lines, "door"), count_stretches(lines, "opening")
            ax.set_title(f"{title}\n{nd} doors, {no} wide openings, closed border", fontsize=14, weight="bold")
            ax.legend([Line2D([], [], color=COLORS[k], lw=3, ls="--" if k == "unseen" else "-") for k in KINDS],
                      ["wall", "door", "wide opening", "unseen (no wall seen)"], loc="lower right", fontsize=10)
    axes[0].set_xlim(box[:2]); axes[0].set_ylim(box[2:])
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def doors_closeup(raw_aligned, lines, res, save_path="doors_closeup.png"):
    """One close-up per door / wide opening on the raw grid, so each can be checked by eye. Saved, not shown."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    gaps = stretches(lines, ("door", "opening"))
    if not gaps:
        return None
    n = len(gaps)
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    cmap = ListedColormap(["#C8C8C8", "#FFFFFF", "#000000"])
    for ax in axes.ravel():
        ax.axis("off")
    for i, (kind, segs) in enumerate(gaps):
        ax = axes.ravel()[i]
        xs_ = [v for l in segs for v in (l["x1"], l["x2"])]
        ys_ = [v for l in segs for v in (l["y1"], l["y2"])]
        cx, cy = (min(xs_) + max(xs_)) / 2, (min(ys_) + max(ys_)) / 2
        span = max(max(xs_) - min(xs_), max(ys_) - min(ys_))
        r = max(int(round(2.0 / res)), int(span * 0.75))
        length = sum(l["length_meters"] for l in segs)
        ax.imshow(raw_aligned, cmap=cmap, vmin=-1, vmax=1, origin="lower", interpolation="nearest")
        _draw(ax, lines, lw=2.2)
        ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy - r, cy + r)
        ax.set_title(f"{i + 1}. {kind} {length:.2f} m", fontsize=10,
                     color=COLORS[kind] if kind == "door" else "#B05E00")
    plt.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    print(f"Saved: {save_path}")
    return save_path


def figures(d, region, streaks, current, lines, show):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    g = d["grid"]
    ys, xs = np.nonzero(g >= 0)
    box = (xs.min() - 5, xs.max() + 5, ys.min() - 5, ys.max() + 5)
    bg = np.where(g == -1, 205, 255).astype(np.uint8)

    # A: what the beam-fan rule sees
    cls = np.zeros(g.shape, np.uint8)   # 0 unknown, 1 solid free, 2 streak, 3 wall
    cls[region] = 1
    cls[streaks] = 2
    cls[g == 1] = 3
    fig, ax = plt.subplots(figsize=(8, 14))
    ax.imshow(cls, cmap=ListedColormap(["#D0D0D0", "#FFFFFF", "#F2A65A", "#000000"]), vmin=0, vmax=3,
              origin="lower", interpolation="nearest")
    ax.set_title("Free space split: white = solid free space, orange = beam streaks")
    ax.set_xlim(box[:2]); ax.set_ylim(box[2:]); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout(); fig.savefig("free_space_split.png", dpi=150)
    if not show:
        plt.close(fig)

    # B: current vectoriser vs outline
    fig, axes = plt.subplots(1, 2, figsize=(18, 14), sharex=True, sharey=True)
    for ax in axes:
        ax.imshow(bg, cmap="gray", vmin=0, vmax=255, origin="lower", interpolation="nearest")
        ax.set_aspect("equal"); ax.axis("off")
    for l in current:
        axes[0].plot([l["x1"], l["x2"]], [l["y1"], l["y2"]], color="black", lw=2.2, solid_capstyle="round")
    _draw(axes[1], lines)
    axes[0].set_title(f"Current vectoriser ({CLEANER})\n{len(current)} segments", fontsize=14, weight="bold")
    nd = count_stretches(lines, "door")
    no = count_stretches(lines, "opening")
    axes[1].set_title(f"Free-space outline + beam-fan doors\n{len(lines)} segments, {nd} doors, {no} wide openings",
                      fontsize=14, weight="bold")
    axes[1].legend([Line2D([], [], color=COLORS[k], lw=3, ls="--" if k == "unseen" else "-") for k in KINDS],
                   ["wall", "door", "wide opening", "unseen (no wall seen)"], loc="lower right", fontsize=11)
    axes[0].set_xlim(box[:2]); axes[0].set_ylim(box[2:])
    plt.tight_layout(); fig.savefig("outline_vs_current.png", dpi=150)
    if not show:
        plt.close(fig)

    # C: every door and opening close up, on the raw grid, so each one can be checked by eye
    doors_closeup(d["raw"], lines, d["res"])
    if show:
        plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    """Standalone run: compares the outline with the current vectoriser in its own folder."""
    import matplotlib
    if not SHOW_PLOTS:
        matplotlib.use("Agg")      # standalone only; main.py's own runs are never affected
    import main as pipeline       # settings (BAG_PATH, weights, output folder) come from main.py
    map_yaml = pipeline.BAG_PATH
    out_dir = os.path.join(pipeline.output_dir_for(map_yaml), "door_outline_prototype")
    os.makedirs(out_dir, exist_ok=True)
    start = os.getcwd()
    os.chdir(out_dir)      # the cleaners save files to the current folder
    try:
        print(f"Output folder: {out_dir}")
        d = load_and_align(map_yaml, pipeline.FPUNET_WEIGHTS_PATH)
        res = d["res"]
        print(f"Rotation (unchanged method): {d['angle']:.2f} degrees")

        lines, polys, region, streaks = outline(d["grid"], d["raw"], res)
        current = current_vector(d["grid"], res, merge=pipeline.MERGE_WALLS)

        raw_wall = d["raw"] == 1
        walls = [l for l in lines if l["kind"] == "wall"]
        p_new, r_new = evidence(walls, raw_wall, res)
        p_cur, r_cur = evidence(current, raw_wall, res)
        c_new, loose_new = connectivity(lines, res)
        c_cur, loose_cur = connectivity(current, res)

        def total(kind):
            return sum(l["length_meters"] for l in lines if l["kind"] == kind)

        report = {
            "map": os.path.basename(map_yaml), "cleaner": CLEANER, "rotation_deg": round(float(d["angle"]), 3),
            "current_vectoriser": {"segments": len(current), "separate_pieces": c_cur, "loose_wall_ends": loose_cur,
                                   "wall_length_m": round(sum(l["length_meters"] for l in current), 1),
                                   "walls_on_scan_data": round(p_cur, 3), "scan_walls_covered": round(r_cur, 3)},
            "outline_prototype": {"segments": len(lines), "separate_pieces": c_new, "loose_wall_ends": loose_new,
                                  "polygons": {k: sum(1 for p in polys if p[0] == k) for k in ("outer", "hole")},
                                  "doors": count_stretches(lines, "door"),
                                  "door_widths_m": sorted(round(sum(l["length_meters"] for l in segs), 2)
                                                          for _, segs in stretches(lines, ("door",))),
                                  "door_straight_widths_m": sorted(
                                      round(float(np.hypot(segs[-1]["x2"] - segs[0]["x1"], segs[-1]["y2"] - segs[0]["y1"])) * res, 2)
                                      for _, segs in stretches(lines, ("door",))),
                                  "wide_openings": count_stretches(lines, "opening"),
                                  "segments_by_kind": {k: sum(l["kind"] == k for l in lines) for k in KINDS},
                                  "length_m": {k: round(total(k), 1) for k in KINDS},
                                  "walls_on_scan_data": round(p_new, 3), "scan_walls_covered": round(r_new, 3)},
            "note": "walls_on_scan_data / scan_walls_covered: share within 0.15 m of raw wall pixels. "
                    "The outline only follows walls that border the explored free space, so walls of rooms "
                    "the robot never entered are not part of it.",
        }
        with open("report.json", "w") as f:
            json.dump(report, f, indent=2)
        export(lines, res, d["M"], d["origin"], d["angle"], "floorplan_outline",
               source_name=os.path.basename(map_yaml), cleaner=CLEANER)
        figures(d, region, streaks, current, lines, SHOW_PLOTS)

        print()
        print("==============================================")
        print("DOOR + OUTLINE PROTOTYPE")
        print("==============================================")
        print(json.dumps(report, indent=1))
        print("Files: floorplan_outline.json / .geojson / .svg, outline_vs_current.png, "
              "doors_closeup.png, free_space_split.png, report.json")
    finally:
        os.chdir(start)


if __name__ == "__main__":
    main()
