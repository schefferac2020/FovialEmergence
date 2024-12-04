import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import random
import numpy as np


SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

class ClutteredMNISTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for Cluttered MNIST.
        :param root_dir: Root directory of the dataset (e.g., "dataset/cluttered_mnist")
        :param transform: Optional torchvision transforms to apply to the images
        """
        self.root_dir = root_dir
        self.transform = transform

        # Gather all image paths and their labels
        self.data = []
        for label in range(10):  # Assuming labels are 0-9
            label_dir = os.path.join(root_dir, str(label))
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    if file_name.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(label_dir, file_name)
                        self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an image and its label at the specified index.
        :param idx: Index of the data point
        :return: Tuple (image, label)
        """
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(batch_size=64):
    '''gets the train and test dataloaders'''
    
    random.seed(SEED)  # Python random module
    np.random.seed(SEED)  # NumPy random module
    torch.manual_seed(SEED)  # PyTorch random module
    torch.cuda.manual_seed_all(SEED)

    # Define transforms for the dataset
    transform = transforms.Compose([
        transforms.Resize((100, 100)),  # Resize images to 100x100
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    # Create dataset
    dataset_dir = "dataset/cluttered_mnist"
    cluttered_mnist_dataset = ClutteredMNISTDataset(root_dir=dataset_dir, transform=transform)

    # Split dataset into train and test (80% train, 20% test)
    train_size = int(0.9 * len(cluttered_mnist_dataset))
    test_size = len(cluttered_mnist_dataset) - train_size
    train_dataset, test_dataset = random_split(cluttered_mnist_dataset, [train_size, test_size])

    # DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

