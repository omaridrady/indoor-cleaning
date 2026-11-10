import json
import numpy as np
import cv2

# ============================================================
# CONFIGURATION (meters)
# ============================================================

# Parallel walls closer than this (sideways) are treated as ONE wall.
MERGE_SIDE_TOLERANCE_M = 0.15

# Two collinear pieces separated by a gap up to this are joined into one wall.
# Doorways are ~0.8-0.9 m, so keep this below that if you still want doors visible
# for the later door-detection stage. 0.5 closes SLAM breaks but keeps doors.
MERGE_GAP_M = 0.5

# A wall shorter than this that touches no other wall is clutter and is dropped.
MIN_ISOLATED_LENGTH_M = 1.0
TOUCH_TOLERANCE_M = 0.3

# Joins snapped endpoints; the same value used in room_polygon_builder.
JSON_DECIMALS = 3


# ============================================================
# MERGING
# ============================================================

def _normalize(lines):
    """Lines -> separate horizontal/vertical lists of {pos, lo, hi}."""
    horiz, vert = [], []
    for l in lines:
        fam = l.get("family")
        if fam not in ("horizontal", "vertical"):
            a = l["angle_degrees"] % 180.0
            fam = "horizontal" if min(a, 180.0 - a) <= 45.0 else "vertical"
        if fam == "horizontal":
            horiz.append({"pos": (l["y1"] + l["y2"]) / 2.0,
                          "lo": min(l["x1"], l["x2"]), "hi": max(l["x1"], l["x2"])})
        else:
            vert.append({"pos": (l["x1"] + l["x2"]) / 2.0,
                         "lo": min(l["y1"], l["y2"]), "hi": max(l["y1"], l["y2"])})
    return horiz, vert


def _merge_family(items, side_tol, gap):
    """
    1) group parallel walls that sit within side_tol of each other,
    2) inside each group, join pieces whose gap is <= gap,
    3) give every merged wall the length-weighted average position.
    """
    if not items:
        return []
    items = sorted(items, key=lambda w: w["pos"])

    # 1) group by sideways position
    groups, cur = [], [items[0]]
    for w in items[1:]:
        mean = np.average([c["pos"] for c in cur], weights=[c["hi"] - c["lo"] + 1e-6 for c in cur])
        if abs(w["pos"] - mean) <= side_tol:
            cur.append(w)
        else:
            groups.append(cur)
            cur = [w]
    groups.append(cur)

    merged = []
    for g in groups:
        g = sorted(g, key=lambda w: w["lo"])
        # 2) chain pieces along the wall direction
        chains = [[g[0]]]
        reach = g[0]["hi"]
        for w in g[1:]:
            if w["lo"] - reach <= gap:
                chains[-1].append(w)
                reach = max(reach, w["hi"])
            else:
                chains.append([w])
                reach = w["hi"]
        # 3) one wall per chain
        for ch in chains:
            weights = [c["hi"] - c["lo"] + 1e-6 for c in ch]
            merged.append({
                "pos": float(np.average([c["pos"] for c in ch], weights=weights)),
                "lo": min(c["lo"] for c in ch),
                "hi": max(c["hi"] for c in ch),
            })
    return merged


def _drop_isolated_stubs(horiz, vert, min_len, touch):
    """Removes short walls that touch nothing (leftover clutter)."""
    def touches(w, own_kind):
        others_same = horiz if own_kind == "h" else vert
        others_perp = vert if own_kind == "h" else horiz
        for p in others_perp:   # perpendicular wall near an end or spanning it
            if p["lo"] - touch <= w["pos"] <= p["hi"] + touch and \
               (w["lo"] - touch <= p["pos"] <= w["hi"] + touch):
                return True
        return False

    def keep(ws, kind):
        out = []
        for w in ws:
            if (w["hi"] - w["lo"]) >= min_len or touches(w, kind):
                out.append(w)
        return out

    return keep(horiz, "h"), keep(vert, "v")


def merge_walls(lines, resolution):
    """
    Cleans up the snapped wall list: joins broken/duplicate walls into long ones
    and removes tiny isolated stubs. Works in the aligned frame (pixels).
    Returns line dicts in the same format as the other stages.
    """
    from room_polygon_builder import snap_wall_endpoints

    side = MERGE_SIDE_TOLERANCE_M / resolution
    gap = MERGE_GAP_M / resolution
    min_len = MIN_ISOLATED_LENGTH_M / resolution
    touch = TOUCH_TOLERANCE_M / resolution

    horiz, vert = _normalize(lines)
    n_before = len(horiz) + len(vert)

    horiz = _merge_family(horiz, side, gap)
    vert = _merge_family(vert, side, gap)
    horiz, vert = _drop_isolated_stubs(horiz, vert, min_len, touch)

    # Rebuild as line dicts, then snap again so corners meet exactly.
    rebuilt = []
    for h in horiz:
        rebuilt.append(_line(h["lo"], h["pos"], h["hi"], h["pos"], "horizontal", resolution))
    for v in vert:
        rebuilt.append(_line(v["pos"], v["lo"], v["pos"], v["hi"], "vertical", resolution))

    final = snap_wall_endpoints(rebuilt, resolution)
    print(f"Merged walls: {n_before} -> {len(final)} segments")
    return final


def _line(x1, y1, x2, y2, family, resolution):
    lp = float(np.hypot(x2 - x1, y2 - y1))
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "length_pixels": lp, "length_meters": lp * resolution,
            "angle_degrees": float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0),
            "family": family}


# ============================================================
# EXPORT
# ============================================================

def _to_meters(lines, resolution):
    """Aligned-frame pixels -> meters, shifted so the drawing starts at (0, 0)."""
    xs = [v for l in lines for v in (l["x1"], l["x2"])]
    ys = [v for l in lines for v in (l["y1"], l["y2"])]
    min_px, min_py = min(xs), min(ys)
    segs = [[(l["x1"] - min_px) * resolution, (l["y1"] - min_py) * resolution,
             (l["x2"] - min_px) * resolution, (l["y2"] - min_py) * resolution] for l in lines]
    width = (max(xs) - min_px) * resolution
    height = (max(ys) - min_py) * resolution
    return segs, width, height, (min_px, min_py)


def _to_map_frame_matrix(align_M, resolution, map_origin, shift_px):
    """
    2x3 matrix: exported meters (x right, y up) -> original SLAM map frame meters.
    Lets the app overlay the robot pose, which is reported in the map frame.
    """
    Minv = cv2.invertAffineTransform(align_M)
    A, t = Minv[:, :2], Minv[:, 2]
    m = np.array(shift_px, dtype=float)
    o = np.array(map_origin, dtype=float)
    return np.hstack([A, (resolution * (A @ m + t) + o).reshape(2, 1)])


def export_floorplan(lines, resolution, align_M, map_origin, rotation_deg,
                     base_name="floorplan", source_name=""):
    """
    Writes three files:
      <base>.json     compact format for the Flutter app (recommended)
      <base>.geojson  standard GeoJSON LineStrings (local meters, not lat/lon)
      <base>.svg      for quick viewing / flutter_svg
    """
    if not lines:
        print("No walls to export.")
        return None

    segs, width, height, shift_px = _to_meters(lines, resolution)
    r = JSON_DECIMALS
    to_map = _to_map_frame_matrix(align_M, resolution, map_origin, shift_px)

    walls = [{"id": i, "x1": round(s[0], r), "y1": round(s[1], r),
              "x2": round(s[2], r), "y2": round(s[3], r),
              "length": round(float(np.hypot(s[2] - s[0], s[3] - s[1])), r)}
             for i, s in enumerate(segs)]

    doc = {
        "format": "floorplan-v1",
        "units": "meters",
        "axes": "x right, y up, origin at the bottom-left of the drawing",
        "source": source_name,
        "width": round(width, r),
        "height": round(height, r),
        "rotation_applied_deg": round(float(rotation_deg), 3),
        "to_map_frame": [[round(float(v), 6) for v in row] for row in to_map],
        "walls": walls,
    }
    with open(f"{base_name}.json", "w") as f:
        json.dump(doc, f, indent=2)

    gj = {"type": "FeatureCollection",
          "properties": {"units": "meters", "note": "local metric coordinates, not WGS84"},
          "features": [{"type": "Feature", "id": w["id"],
                        "properties": {"kind": "wall", "length_m": w["length"]},
                        "geometry": {"type": "LineString",
                                     "coordinates": [[w["x1"], w["y1"]], [w["x2"], w["y2"]]]}}
                       for w in walls]}
    with open(f"{base_name}.geojson", "w") as f:
        json.dump(gj, f)

    # SVG: y axis points down, so flip y.
    pad = 0.5
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'viewBox="{-pad} {-pad} {width + 2 * pad:.3f} {height + 2 * pad:.3f}">',
             f'<rect x="{-pad}" y="{-pad}" width="{width + 2 * pad:.3f}" '
             f'height="{height + 2 * pad:.3f}" fill="white"/>',
             '<g stroke="black" stroke-width="0.12" stroke-linecap="round">']
    for w in walls:
        parts.append(f'<line x1="{w["x1"]}" y1="{height - w["y1"]:.3f}" '
                     f'x2="{w["x2"]}" y2="{height - w["y2"]:.3f}"/>')
    parts += ['</g>', '</svg>']
    with open(f"{base_name}.svg", "w") as f:
        f.write("\n".join(parts))

    print(f"Exported {len(walls)} walls ({width:.1f} x {height:.1f} m): "
          f"{base_name}.json / .geojson / .svg")
    return doc
