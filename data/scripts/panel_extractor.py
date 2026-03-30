"""
OpenCV contour-based comic panel extractor.
"""
import cv2
import os


def extract_panels(img_path, out_dir, min_area=8000):
    """
    Extract individual panels from a full comic page image.
    Saves panel crops + a debug overlay image.
    Returns the number of panels extracted.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ⚠  Could not read: {img_path}")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection + dilation
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dil = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    base = os.path.splitext(os.path.basename(img_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    pid = 0
    vis = img.copy()

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        panel = img[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(out_dir, f"{base}_panel_{pid}.jpg"), panel)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pid += 1

    # Debug overlay
    cv2.imwrite(os.path.join(out_dir, f"{base}_debug.jpg"), vis)
    return pid
