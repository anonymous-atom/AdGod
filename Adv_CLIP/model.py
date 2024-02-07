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
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_model():
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
    return model

def get_processor():
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    return processor