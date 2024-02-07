import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from model import get_model
from dataset import get_train_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import transforms
from torch.distributed import init_process_group, destroy_process_group
from PIL import Image
import os
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--local-rank", default=0, type=int)
args = parser.parse_args()

torch.distributed.init_process_group('nccl')

model = get_model()
dataset = get_train_data()

dist_sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=64,  sampler=dist_sampler)

device = torch.device('cuda', args.local_rank)
model = model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model,  device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

accumulation_steps = 20

for epoch in range(20):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    accumulated_loss = 0.0
    
    for batch_idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(dim=1)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(dim=1)
        inputs['input_ids'] = inputs['input_ids'].squeeze(dim=0)
        outputs = model(**inputs)
        
        loss = criterion(outputs.logits_per_image, labels)
        running_loss += loss.item()
        accumulated_loss += loss.item()
        
        loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        _, predicted = torch.max(outputs.logits_per_image, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
    
    if batch_idx % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1} loss: {epoch_loss:.4f}, accuracy: {epoch_accuracy:.2f}')
    torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')
    print(f"Model saved after epoch {epoch + 1}.")