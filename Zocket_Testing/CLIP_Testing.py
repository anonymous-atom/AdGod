#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
import requests
import time
from Adv_CLIP.model import get_model, get_processor

model = get_model()
model.to("cuda")
processor = get_processor()
# Print on which device the model is
print(model.device)


# ## Checking out CLIP

# In[10]:


# Create function to pass input image and text to model and return the label probabilities
import torch

def get_label_probs(image, text, model, processor):
    torch.cuda.empty_cache()  # Release cached memory
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    inputs = inputs.to("cuda")
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    # Clear GPU memory
    del inputs, outputs, logits_per_image
    torch.cuda.empty_cache()  # Release cached memory
    return probs


im = "dog.jpg"
image = Image.open(im)
text=["a photo of a cat", "a photo of a dog"]

# Get label probabilities for the image and text
start = time.time()
probs = get_label_probs(image, text, model, processor)
end = time.time()
print(f"Time taken: {end-start}")
print(probs)


# In[11]:


# Load JSON file with image URLs and text
import json
with open("zocket_brand_ads.json", "r") as f:
    data = json.load(f)

print(data['entries'][0])


# In[12]:


# Extract entries which have "https://cdn.shopify.com/" as part of the image URL
entries = [entry for entry in data['entries'] if "https://cdn.shopify.com/" in entry['url']]
len(entries)


# In[13]:


import matplotlib.pyplot as plt

# Load image from URL
print(entries[3]['url'])
image = Image.open(requests.get(entries[3]['url'], stream=True).raw)
# Plot image
plt.imshow(image)


# In[17]:


image = Image.open(requests.get(entries[3]['url'], stream=True).raw)
text = ['Ad Creative with text', 'Ad Creative with No text']

output = get_label_probs(image, text, model, processor)

output


# In[18]:


torch.cuda.empty_cache()


# ## Zocket Generated Image, Grouping the entries by brand

# In[9]:


from collections import defaultdict
import json

grouped_entries = defaultdict(lambda: {"name": "", "brand_name": "", "brand_logo": "", "products": []})

for entry in entries:
    brand_name = entry['brand_name']
    grouped_entries[brand_name]['name'] = entry['name']
    grouped_entries[brand_name]['brand_name'] = brand_name
    grouped_entries[brand_name]['brand_logo'] = entry['brand_logo']
    product_info = {"product_name": entry["product_name"], "url": entry["url"]}
    grouped_entries[brand_name]['products'].append(product_info)

# Convert defaultdict to list of dictionaries with products
grouped_entries_list = list(grouped_entries.values())

# Print or use grouped_entries_list as needed

# Save grouped_entries_list to a JSON file
with open('zocket_grouped_entries_2.json', 'w') as json_file:
    json.dump(grouped_entries_list, json_file, indent=2)


# In[10]:


len(grouped_entries_list[1]['products'])

total = 0
out_a = []
for i in range(len(grouped_entries_list)):
    out_a.append([grouped_entries_list[i]['brand_name'], len(grouped_entries_list[i]['products'])])
    total += len(grouped_entries_list[i]['products'])

print(total, out_a)


# In[11]:


sorted_data = sorted(out_a, key=lambda x: x[1], reverse=True)

print(sorted_data)


# In[12]:


all_img_urls = []
for i in range(253):
    for z in  range(len(grouped_entries_list[i]['products'])):
        all_img_urls.append({'product_name': grouped_entries_list[i]['products'][z]['product_name'], 'url': grouped_entries_list[i]['products'][z]['url']})


# In[13]:


print(len(all_img_urls),"\n", all_img_urls[0])


# In[14]:


import random
rand_100_indx = random.sample(range(0, len(all_img_urls)), 100)
rand_100_img_urls = [all_img_urls[i] for i in rand_100_indx]

rand_100_img_urls


# In[16]:


import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
for i in tqdm(range(len(rand_100_img_urls))):
    print(rand_100_img_urls[i]['url'])
    image = Image.open(requests.get(rand_100_img_urls[i]['url'], stream=True).raw)
    text = ['Ad Creative with text', 'Ad Creative with No text']
    output_label_probs = get_label_probs(image, text)

    # Create plot with 100 images, product name and label probabilities
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Product Name: {rand_100_img_urls[i]['product_name']}\nLabel Probabilities: {output_label_probs}")
    plt.show()


# In[ ]:




