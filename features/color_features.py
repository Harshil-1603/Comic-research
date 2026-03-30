import cv2, numpy as np, torch


def hsv_hist(image, bins=16):
    """Standard HSV histogram feature extraction"""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 255))[0]
    v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 255))[0]
    feat = np.concatenate([h, s, v]).astype("float32")
    feat /= feat.sum() + 1e-6
    return torch.tensor(feat)  # (48,)


def hsv_hist_background(image, bins=16, k=3):
    """
    Background-aware HSV histogram using k-means clustering.
    Treats largest cluster as background and computes HSV on that cluster only.
    """
    # Reshape image to pixels
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find largest cluster (background)
    cluster_sizes = np.bincount(labels.flatten())
    bg_cluster = np.argmax(cluster_sizes)
    
    # Get background pixels
    bg_mask = (labels == bg_cluster).reshape(image.shape[:2])
    bg_pixels = image[bg_mask]
    
    if len(bg_pixels) == 0:
        # Fallback to standard histogram
        return hsv_hist(image, bins)
    
    # Compute HSV histogram on background only
    hsv_bg = cv2.cvtColor(bg_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = np.histogram(hsv_bg[:, 0], bins=bins, range=(0, 180))[0]
    s = np.histogram(hsv_bg[:, 1], bins=bins, range=(0, 255))[0]
    v = np.histogram(hsv_bg[:, 2], bins=bins, range=(0, 255))[0]
    
    feat = np.concatenate([h, s, v]).astype("float32")
    feat /= feat.sum() + 1e-6
    return torch.tensor(feat)  # (48,)
