"""
Comic panel extractor using thick-border rectangle detection.

Strategy:
  1. Convert to grayscale → threshold → morphological closing
     (closes gaps in thick panel borders without losing the border itself)
  2. Find external contours on the closed binary image
  3. Keep only contours that:
       - Have a tight bounding-rect (high fill ratio → rectangle, not art)
       - Are larger than MIN_PAGE_FRACTION of the page area
       - Have a reasonable aspect ratio (not a thin line or a sliver)
       - Don't overlap heavily with an already-accepted panel (dedup)
  4. Limit to MAX_PANELS_PER_PAGE (safety cap)

This reliably finds 4-10 panels/page in standard comic layouts.
"""
import cv2
import numpy as np
import os


# ── Tunable parameters ────────────────────────────────────────────────────────

MIN_PAGE_FRACTION = 0.02    # panel must be ≥ 2% of page area
MAX_PAGE_FRACTION = 0.90    # panel must be < 90% of page area (skip full-page splash)
MIN_FILL_RATIO    = 0.60    # bounding-rect fill: contour area / bounding-rect area
MIN_ASPECT        = 0.15    # width/height ≥ 0.15  (not a vertical sliver)
MAX_ASPECT        = 6.5     # width/height ≤ 6.5   (not a horizontal strip)
MAX_PANELS        = 12      # hard cap per page
IOU_THRESHOLD     = 0.30    # suppress duplicate/nested boxes above this IoU


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iou(a, b):
    """Intersection-over-Union for (x,y,w,h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx);  iy = max(ay, by)
    ix2 = min(ax+aw, bx+bw);  iy2 = min(ay+ah, by+bh)
    inter = max(0, ix2-ix) * max(0, iy2-iy)
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, threshold=IOU_THRESHOLD):
    """Non-maximum suppression: keep larger box when two overlap heavily."""
    boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)  # largest first
    kept = []
    for box in boxes:
        if all(_iou(box, k) < threshold for k in kept):
            kept.append(box)
    return kept


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_panels(img_path, out_dir, debug=True):
    """
    Extract comic panels delimited by thick rectangular borders.

    Args:
        img_path : path to the full-page JPEG
        out_dir  : directory to write panel crops + debug image
        debug    : save a debug overlay image

    Returns:
        int — number of panels saved
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠  Could not read: {img_path}")
        return 0

    H, W = img.shape[:2]
    page_area = H * W

    # ── 1. Pre-process ────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold handles varying paper brightness / yellowing
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4,
    )

    # Morphological closing: bridges the thick panel borders into solid walls.
    # Kernel size ~1% of the shorter page dimension works well for 150 DPI scans.
    k = max(5, min(W, H) // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # ── 2. Find contours ──────────────────────────────────────────────────────
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ── 3. Filter candidates ──────────────────────────────────────────────────
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area     = w * h
        c_area   = cv2.contourArea(c)

        # Skip if too small or too large relative to page
        if area < MIN_PAGE_FRACTION * page_area:
            continue
        if area > MAX_PAGE_FRACTION * page_area:
            continue

        # Skip if the contour doesn't fill its bounding rect well
        # (a panel border fills its rect; irregular art shapes don't)
        fill = c_area / area if area > 0 else 0
        if fill < MIN_FILL_RATIO:
            # Relax fill check: some panels have intricate border art.
            # Instead check approximate rectangularity via hull vs bbox.
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            hull_fill = hull_area / area if area > 0 else 0
            if hull_fill < MIN_FILL_RATIO:
                continue

        # Aspect ratio check
        aspect = w / h if h > 0 else 0
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        candidates.append((x, y, w, h))

    # ── 4. Deduplicate overlapping boxes ─────────────────────────────────────
    panels = _nms(candidates, IOU_THRESHOLD)

    # Sort panels reading-order: top-to-bottom, left-to-right
    panels.sort(key=lambda b: (b[1] // (H // 6), b[0]))

    # Hard cap
    panels = panels[:MAX_PANELS]

    # ── 5. Save crops ─────────────────────────────────────────────────────────
    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    vis = img.copy()

    for pid, (x, y, w, h) in enumerate(panels):
        crop = img[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_dir, f"{base}_panel_{pid}.jpg"), crop)
        if debug:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 220, 0), 3)
            cv2.putText(vis, str(pid), (x+6, y+28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)

    if debug:
        cv2.imwrite(os.path.join(out_dir, f"{base}_debug.jpg"), vis)

    return len(panels)
