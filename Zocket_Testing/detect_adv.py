import cv2
import numpy as np
from collections import Counter
import pytesseract

from PIL import Image


from paddleocr import PaddleOCR, draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
def detect_text(image):

    result = ocr.ocr(image, cls=True)
    txt = ''
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            txt += line[1][0]

    return txt
def analyze_text(text):
    marketing_keywords = ['sale', 'offer', 'discount', 'promotion', 'limited', 'buy', 'now', ]

    # Count the occurrences of marketing keywords
    word_count = Counter([word.lower() for word in text.split()])
    keyword_count = sum(word_count[keyword] for keyword in marketing_keywords)

    # Classify based on the number of marketing keywords
    if keyword_count > 2:
        return "Advertisement"
    else:
        return "Normal Product Image"


# Point 2: Layout and Composition Analysis

def analyze_layout(image_path):

    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert the grayscale image
    inverted = cv2.bitwise_not(gray)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counters
    asymmetric_count = 0
    dynamic_shape_count = 0

    # Iterate through contours
    for contour in contours:
        # Calculate the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate aspect ratio
        aspect_ratio = float(w) / h

        # Check for asymmetric layout
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            asymmetric_count += 1

        # Check for dynamic shape
        if len(contour) > 5:
            _, _, angle = cv2.fitEllipse(contour)
            if angle > 30 and angle < 150:
                dynamic_shape_count += 1

    # Determine if it's an advertisement based on criteria
    is_advertisement = False
    if asymmetric_count > 1 or dynamic_shape_count > 1:
        is_advertisement = True

    return is_advertisement





# Point 3: Color Analysis
def analyze_color(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate mean saturation and value
    mean_saturation = np.mean(hsv_image[:, :, 1])
    mean_value = np.mean(hsv_image[:, :, 2])

    # Check for high saturation and value (vivid colors)
    if mean_saturation > 150 and mean_value > 150:
        return "Advertisement"
    else:
        return "Not Advertisement"



# Point 4: Edge Detection and Shape Analysis
def analyze_shapes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for specific shapes (e.g., arrows, starbursts)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) in [3, 5, 7]:  # Triangles, pentagons, or starbursts
            return True

    return False


# # Load the image
# image = '/home/karun/PycharmProjects/AdGod/250.jpg'
# img_open = Image.open(image)
#
# # Analyze the image using different techniques
# text_result = analyze_text(detect_text(image))
# layout_result = analyze_layout(image)
# color_result = analyze_color(image)
# shape_result = analyze_shapes(image)
#
# # Print the results
# print("Text Analysis Result:", text_result)
# print("Layout Analysis Result:", layout_result)
# print("Shape Analysis Result:", shape_result)