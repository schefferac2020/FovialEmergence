import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the RNN model
class MNISTRNN(nn.Module):
    def __init__(self, crop_size, hidden_size, num_layers, num_classes):
        super(MNISTRNN, self).__init__()
        self.crop_size = crop_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN to process the crops
        self.rnn = nn.RNN(crop_size**2, hidden_size, num_layers, batch_first=True)
        self.fc_class = nn.Linear(hidden_size, num_classes)  # Class prediction
        self.fc_action = nn.Linear(hidden_size, 2)  # Next crop center prediction

    def crop(self, padded_image, center):
        """
        Crop a region around the given center from the padded image.
        :param padded_image: Padded input image (batch_size, 1, padded_size, padded_size)
        :param center: Crop centers (batch_size, 2)
        :return: Cropped image regions (batch_size, crop_size, crop_size)
        """
        crop_size = self.crop_size
        half_crop = crop_size // 2

        # Compute cropping indices
        x_start = (center[:, 0] - half_crop).long()
        x_end = (center[:, 0] + half_crop).long()
        y_start = (center[:, 1] - half_crop).long()
        y_end = (center[:, 1] + half_crop).long()

        # Perform efficient tensor slicing
        crops = torch.stack([
            padded_image[b, :, y_start[b].item():y_end[b].item(), x_start[b].item():x_end[b].item()]
            for b in range(padded_image.size(0))
        ])
        return crops

    def forward(self, image, center, h0):
        """
        Forward pass with dynamic cropping and RNN processing.
        :param image: Full input image (batch_size, 1, 28, 28)
        :param center: Initial crop centers (batch_size, 2)
        :param h0: Initial hidden state (num_layers, batch_size, hidden_size)
        :return: Class prediction, next center, hidden state
        """
        batch_size = image.size(0)
        crop_size = self.crop_size

        # Pad the image to avoid boundary issues
        padding = crop_size // 2
        padded_image = nn.functional.pad(image, (padding, padding, padding, padding), mode='constant', value=0)

        # Extract crops
        crops = self.crop(padded_image, center)  # (batch_size, 1, crop_size, crop_size)
        crops = crops.view(batch_size, -1)  # Flatten the crop (batch_size, crop_size^2)

        # Process with RNN
        crops = crops.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, crop_size^2)
        out, hn = self.rnn(crops, h0)

        # Predict class and next crop center
        class_pred = self.fc_class(out[:, -1, :])  # Class prediction
        action_pred = self.fc_action(out[:, -1, :])  # Action (next crop center)

        return class_pred, action_pred, hn

# Hyperparameters
crop_size = 20
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 64
learning_rate = 0.001
num_epochs = 10
num_steps = 10  # RNN steps per image

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./dataset/raw_mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create model, optimizer, and loss functions
model = MNISTRNN(crop_size, hidden_size, num_layers, num_classes)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion_class = nn.CrossEntropyLoss()
criterion_action = nn.MSELoss()  # For predicting the next center

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        batch_size = images.size(0)

        # Initialize the hidden state and center
        h0 = torch.zeros(num_layers, batch_size, hidden_size).to(images.device)
        centers = torch.tensor([[14, 14]] * batch_size).float().to(images.device)  # Initial center

        loss = 0
        for step in range(num_steps):
            # Forward pass
            class_pred, action_pred, h0 = model(images, centers, h0)

            # Compute losses
            loss_class = criterion_class(class_pred, labels)
            # loss_action = criterion_action(action_pred, centers)  # Target is to stay at the initial center
            loss += loss_class

            # Update centers for the next step
            centers = torch.clip(action_pred, min=crop_size // 2, max=28 + crop_size // 2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        batch_size = images.size(0)
        h0 = torch.zeros(num_layers, batch_size, hidden_size).to(images.device)
        centers = torch.tensor([[14, 14]] * batch_size).float().to(images.device)

        for step in range(num_steps):
            class_pred, action_pred, h0 = model(images, centers, h0)
            centers = torch.clip(action_pred, min=crop_size // 2, max=28 + crop_size // 2)

        _, predicted = torch.max(class_pred.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the train images: {100 * correct / total:.2f}%')
