import os

from PIL import Image
import requests
# Create function to pass input image and text to model and return the label probabilities
import torch
import time
from detect_adv import detect_text, analyze_layout, analyze_shapes
from transformers import CLIPProcessor, CLIPModel
# Streamlit code to upload image and output label probabilities
import streamlit as st
import tempfile

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_label_probs(image, text, model, processor):
    torch.cuda.empty_cache()  # Release cached memory
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    inputs = inputs
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # Clear GPU memory
    torch.cuda.empty_cache()
    del inputs, outputs, logits_per_image
    return probs

text = ['Advertisement Creative(Contains Text)', 'Not an Advertisement Creative(Contains No Text)', 'Simple Product Image and not an Advertisement)']



st.title("Advertisement Detection using CLIP")

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type="jpg")


if uploaded_image is not None:
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, uploaded_image.name)
    with open(path, "wb") as f:
        f.write(uploaded_image.getvalue())

    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    # Get label probabilities
    probs = get_label_probs(image, text, model, processor)
    # Output label probabilities
    prob = probs.tolist()
    prob = prob[0]
    # st.write("Label Probabilities:", prob)
    # st.write("Label Probabilities:", probs)
    # # Output predicted label
    # predicted_label = text[torch.argmax(probs[0])]
    # st.write("Predicted Label:", predicted_label)

    # Augmenting using classic techniques
    layout_result = analyze_layout(path)
    shape_result = analyze_shapes(path)
    #
    # # Output classic technique results
    # st.write("Layout Analysis Result:", layout_result)
    # st.write("Shape Analysis Result:", shape_result)
    final_out = False
    # Find index of max value from list
    max_index = prob.index(max(prob))
    if max_index == 0 and (layout_result == True or shape_result == True):
        final_out = True
    # Write 'Advertisement' if the image is an advertisement
    if final_out == True:
        st.write("Advertisement")
    else:
        st.write("Not an Advertisement")

