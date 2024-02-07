from model import get_processor
import json
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


processor = get_processor()

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        text = 'Ad Creative'
        label = 0
        inputs_dict = processor(text=["Ad Creative"], images=img, return_tensors="pt", padding=True)
        return inputs_dict, label



with open('image/QA_Action.json', 'r') as f:
    data = json.load(f)

dataset = []
folder = 'images/'

for image, action in data.items():
  dataset.append({'image_path': folder+image,
                  'title' : action[0]})


def get_train_data():
    train_dataset = CustomDataset(dataset[:25000])
    return train_dataset