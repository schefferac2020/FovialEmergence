import torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from glimpse import GlimpseModel, GlimpseModelFaster
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import matplotlib.pyplot as plt
from datetime import datetime
import logging
from glimpse import plot_glimpse_image

import random
from dataloaders import get_dataloaders
import rnn_model
from rnn_model import MNISTRNN

# Check if CUDA is available and if PyTorch is using GPU
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=1, abbreviated=False))
    
# Load the dataloaders
train_loader, test_loader = get_dataloaders()

# Example: Iterate through the DataLoader
for images, labels in train_loader:
    print("Training batch of images shape:", images.shape)  # (batch_size, 1, 100, 100)
    print("Training batch of labels shape:", labels.shape)  # (batch_size,)
    break

for images, labels in test_loader:
    print("Test batch of images shape:", images.shape)  # (batch_size, 1, 100, 100)
    print("Test batch of labels shape:", labels.shape)  # (batch_size,)
    break

# Hyperparameters
image_size = 100
hidden_size = 512
num_layers = 2
num_classes = 10
batch_size = 64
learning_rate = 0.0002
num_epochs = 100
num_steps = 3  # RNN steps per image
num_kernels = 12*12

# Create model, optimizer, and loss functions
model = MNISTRNN(image_size, hidden_size, num_layers, num_classes, num_kernels, device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_class = nn.CrossEntropyLoss()
criterion_action = nn.MSELoss()  # For predicting the next center


### TRAINING LOOP ###
create_movie = True

# Create model folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

models_path = os.path.join("models", timestamp)
os.makedirs(models_path)
print("Created New Folder:", models_path)

# Initialize Logging
log_file_path = os.path.join(models_path, "training_log.txt")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()  # Print to console
    ]
)

if create_movie:
    video_path = os.path.join("video_frames", timestamp)
    print("Created New Folder:", video_path)
    os.makedirs(video_path)
    vid_background_img = torch.zeros((100, 100))
    
    video_frame = 0

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        batch_size = images.size(0)

        # Initialize the hidden state and center
        h0 = torch.zeros(num_layers, batch_size, hidden_size).to(images.device)
        
        next_actions = torch.zeros((batch_size, 2), device=images.device)

        optimizer.zero_grad()

        loss = 0
        for step in range(num_steps):
            # Print the image out
            
            # next_actions = torch.zeros_like(next_actions, device=next_actions.device) #TODO: remove this line please
            # Forward pass    
            class_pred, action_pred, h0, sensor_readings = model(images, next_actions, h0)
            
            if (i + 1) % 100 == 0:
                sz = torch.ones((1), device=device)
                img = images[0][0]
                sc = next_actions[0] 
                img_name = f"pictures/iter{i+1}_step{step}.png"
                plot_glimpse_image(img_name, img, model.eyes.mu, model.eyes.sigma, sc, sz, sensor_readings[0])
                
            
            # Compute losses
            loss_class = criterion_class(class_pred, labels)
            loss += loss_class
            
            
            next_actions = action_pred
            
            # print(f"step{step}: memory update")
            # print("", torch.cuda.memory_summary(device='cuda:0'))

        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(class_pred.data, 1)
        
        accuracy = (predicted == labels).sum() / labels.size(0)

        if (i + 1) % 100 == 0:
            log_message = (f"Epoch [{epoch + 1}/{num_epochs}], "
               f"Step [{i + 1}/{len(train_loader)}], "
               f"Loss: {loss.item():.4f}, "
               f"Accuracy: {accuracy.item():.4f}")

            logging.info(log_message)
            
            # create movie
            if create_movie:
                frame_file = os.path.join(video_path, f"frame_{video_frame}.png")
                plot_glimpse_image(frame_file, vid_background_img, model.eyes.mu, model.eyes.sigma, torch.zeros((2), device=device), sz, torch.zeros(144))
                video_frame += 1
    
    # Save the model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        # 'loss': 0.25,  # Example loss
    }

    # Save the checkpoint
    checkpoint_path = os.path.join(models_path, f"checkpoint_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    