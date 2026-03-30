"""
Improved OCR module with Gaussian blur + Otsu thresholding.

Usage:
    from features.ocr import extract_text
    text = extract_text(image)   # image: BGR or RGB numpy array
"""
import cv2
import pytesseract


def extract_text(image):
    """
    Extract text from a comic panel image using OCR.

    Args:
        image: numpy array (BGR or RGB — works for both since we go to gray)

    Returns:
        Extracted text string (stripped). Empty string if nothing found.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur reduces noise and improves OCR
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu binarisation — automatically finds optimal threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    text = pytesseract.image_to_string(thresh)
    return text.strip()
