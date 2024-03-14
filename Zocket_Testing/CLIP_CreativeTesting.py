from PIL import Image
import requests
import time
from Adv_CLIP.model import get_model, get_processor

model = get_model()
model.to("cuda")
processor = get_processor()
# Print on which device the model is
print(model.device)

# Create function to pass input image and text to model and return the label probabilities
def get_label_probs(url, text):
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    inputs = inputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
text=["a photo of a cat", "a photo of a dog"]

# Get label probabilities for the image and text
start = time.time()
probs = get_label_probs(url, text)
end = time.time()
print(f"Time taken: {end-start}")
print(probs)