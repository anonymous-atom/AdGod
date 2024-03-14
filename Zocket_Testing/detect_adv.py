import cv2
import numpy as np

def count_objects(img):
    """
    Counts the number of distinct objects in the image using contour detection.
    """
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def detect_text(img):
    """
    Detects the presence of text in the image using the EAST text detector.
    """
    east = cv2.EAST_create()
    boxes = east.detectMultiScale(img)
    return len(boxes) > 0

def measure_background_complexity(img):
    """
    Measures the complexity of the background using edge detection and entropy.
    """
    edges = cv2.Canny(img, 100, 200)
    entropy = np.sum(edges * np.log2(1 + edges)) / (edges.shape[0] * edges.shape[1])
    return entropy

# Example usage
image_path = "image1.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

num_objects = count_objects(img)
text_present = detect_text(img)
background_complexity = measure_background_complexity(img)

print("Number of objects:", num_objects)
print("Text present:", text_present)
print("Background complexity:", background_complexity)