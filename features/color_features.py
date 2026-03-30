import cv2, numpy as np, torch


def hsv_hist(image, bins=16):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 180))[0]
    s = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 255))[0]
    v = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 255))[0]
    feat = np.concatenate([h, s, v]).astype("float32")
    feat /= feat.sum() + 1e-6
    return torch.tensor(feat)  # (48,)
