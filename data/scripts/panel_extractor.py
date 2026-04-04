"""
Comic panel extractor — hybrid whitespace-gutter + contour detection.

Primary method (works for Invincible and modern comics):
  - Finds horizontal WHITE bands between panel rows
  - For each row, finds vertical WHITE bands between panels
  - Correctly handles mixed row widths (one panel wide vs two panels wide)

Fallback method (for classic comics with thick dark borders):
  - Adaptive threshold + morphological closing
  - Contour fill-ratio filtering + IoU NMS

Works on any comic regardless of border style.
"""
import cv2
import numpy as np
import os

# ── Tunable parameters ────────────────────────────────────────────────────────

MAX_PANELS        = 12    # hard cap per page (safety)
MIN_PANEL_FRAC    = 0.03  # panel must be ≥ 3% of page area
WHITE_THRESH      = 220   # pixel brightness ≥ this → "white"  (raised to avoid grey artwork)
GUTTER_COVERAGE   = 0.85  # ≥ 85% of row/col must be white → gutter
                          # (a real gutter spans full page width; speech-bubble rows don't)
MIN_GUTTER_PX     = 5     # gutter must be ≥ 5px wide
MERGE_GUTTER_PX   = 25    # merge gutters within 25px of each other
EDGE_MARGIN_FRAC  = 0.03  # ignore gutters within 3% of page edge
MIN_DIM_FRAC      = 0.04  # skip bands thinner than 4% of page dimension


# ── Projection helpers ────────────────────────────────────────────────────────

def _find_gutters(projection, threshold, min_width, merge_dist, total, edge_frac):
    """
    Find contiguous white bands from a 1D brightness projection.

    Args:
        projection : 1-D numpy array — fraction of white pixels per row/col
        threshold  : minimum fraction to be considered white
        min_width  : minimum run length (pixels) to keep
        merge_dist : merge consecutive runs closer than this
        total      : length of the axis (H or W)
        edge_frac  : ignore runs within this fraction of the axis edges

    Returns:
        list of [start, end] pairs (pixel positions)
    """
    N     = len(projection)
    edge  = int(N * edge_frac)
    white = projection >= threshold

    # Find contiguous white runs
    runs = []
    in_run = False
    start  = 0
    for i in range(N):
        if white[i] and not in_run:
            start  = i
            in_run = True
        elif not white[i] and in_run:
            # Filter: wide enough, not at page edge
            if (i - start >= min_width) and (start >= edge) and (i <= N - edge):
                runs.append([start, i])
            in_run = False
    if in_run and (N - start >= min_width) and (start >= edge):
        runs.append([start, N])

    if not runs:
        return []

    # Merge nearby runs
    merged = [runs[0][:]]
    for r in runs[1:]:
        if r[0] - merged[-1][1] <= merge_dist:
            merged[-1][1] = r[1]
        else:
            merged.append(r[:])

    return merged


def _bands_from_gutters(gutters, total):
    """Return content bands (non-gutter regions), given list of [start,end] gutters."""
    bands = []
    prev  = 0
    for (g0, g1) in gutters:
        if g0 > prev:
            bands.append((prev, g0))
        prev = g1
    if prev < total:
        bands.append((prev, total))
    return bands


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_panels(img_path, out_dir, debug=True):
    """
    Detect and save individual comic panels from a full-page image.

    Returns the number of panel images saved.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠  Could not read: {img_path}")
        return 0

    H, W    = img.shape[:2]
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_area = MIN_PANEL_FRAC * H * W

    # ── Method 1: White-gutter detection ─────────────────────────────────────
    # Works for modern comics (Invincible, Saga, TMNT, etc.) that use
    # white/light gaps between panels rather than thick drawn borders.

    white    = (gray >= WHITE_THRESH).astype(np.float32)
    row_proj = white.mean(axis=1)   # (H,) — per-row white fraction

    h_gutters = _find_gutters(
        row_proj, GUTTER_COVERAGE, MIN_GUTTER_PX, MERGE_GUTTER_PX, H, EDGE_MARGIN_FRAC
    )
    row_bands = _bands_from_gutters(h_gutters, H)

    panels = []
    for (y1, y2) in row_bands:
        if (y2 - y1) < H * MIN_DIM_FRAC:
            continue    # skip hairline-thin horizontal strips

        # Within this row, find vertical gutters
        row_slice = gray[y1:y2, :]
        col_white = (row_slice >= WHITE_THRESH).astype(np.float32)
        col_proj  = col_white.mean(axis=0)   # (W,) — per-col white fraction

        v_gutters = _find_gutters(
            col_proj, GUTTER_COVERAGE, MIN_GUTTER_PX, MERGE_GUTTER_PX, W, EDGE_MARGIN_FRAC
        )
        col_bands = _bands_from_gutters(v_gutters, W)

        for (x1, x2) in col_bands:
            if (x2 - x1) < W * MIN_DIM_FRAC:
                continue   # skip hairline-thin vertical strips
            bw = x2 - x1
            bh = y2 - y1
            if bw * bh >= min_area:
                panels.append((x1, y1, bw, bh))

    # ── Method 2: Contour fallback (thick-border comics) ─────────────────────
    if len(panels) < 2:
        panels = _contour_panels(gray, H, W, min_area)

    # ── Sort reading order + hard cap ─────────────────────────────────────────
    row_height = max(1, H // 6)
    panels.sort(key=lambda b: (b[1] // row_height, b[0]))
    panels = panels[:MAX_PANELS]

    # ── Save panel crops ──────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    vis = img.copy()

    for pid, (x, y, w, h) in enumerate(panels):
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_dir, f"{base}_panel_{pid}.jpg"), crop)
        if debug:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 220, 0), 3)
            cv2.putText(vis, str(pid), (x + 6, y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)

    if debug:
        cv2.imwrite(os.path.join(out_dir, f"{base}_debug.jpg"), vis)

    return len(panels)


# ── Contour fallback ──────────────────────────────────────────────────────────

def _contour_panels(gray, H, W, min_area):
    """
    Fallback for comics with explicit thick dark borders (classic style).
    Uses adaptive threshold + morphological closing to isolate panel rectangles.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )
    k      = max(5, min(W, H) // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    page_area  = H * W
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area  = w * h
        if area < min_area or area > 0.90 * page_area:
            continue
        asp = w / h if h > 0 else 0
        if asp < 0.12 or asp > 7.0:
            continue
        c_area = cv2.contourArea(c)
        if c_area / area < 0.45:
            continue
        candidates.append((x, y, w, h))

    return _nms(candidates)


# ── IoU deduplication ─────────────────────────────────────────────────────────

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix  = max(ax, bx);        iy  = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, threshold=0.30):
    """Non-maximum suppression — remove heavily-overlapping boxes."""
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
    kept  = []
    for box in boxes:
        if all(_iou(box, k) < threshold for k in kept):
            kept.append(box)
    return kept
