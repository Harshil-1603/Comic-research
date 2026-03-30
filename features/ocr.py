import pytesseract, cv2


def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return pytesseract.image_to_string(gray)
