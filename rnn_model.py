import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from glimpse import GlimpseModel, GlimpseModelFaster

"""print('the code god was here')"""
# Define the RNN model
class MNISTRNN(nn.Module):
    def __init__(self, image_size, hidden_size, num_layers, num_classes, num_kernels, device="cuda:0"):
        super(MNISTRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.image_size = image_size
        self.num_kernels = num_kernels
        self.device = device

        # RNN to process the crops
        self.rnn = nn.RNN(num_kernels, hidden_size, num_layers, batch_first=True)
        self.fc_class = nn.Linear(hidden_size, num_classes)  # Class prediction
        self.fc_action = nn.Linear(hidden_size, 3)  # Next crop center prediction #TODO: might have to change this back to 2 for older models than Dec 3
        
        self.eyes = GlimpseModel((image_size, image_size), num_kernels, device=device)
        

    def forward(self, images, actions, h0, print_tensor=False):
        """
        Forward pass with dynamic cropping and RNN processing.
        :param image: Full input image (batch_size, 1, 28, 28)
        :param center: Initial crop centers (batch_size, 2)
        :param h0: Initial hidden state (num_layers, batch_size, hidden_size)
        :return: Class prediction, next center, hidden state
        """
        batch_size = len(images)
        
        
        # sz = torch.ones((batch_size, 1), device=self.device)
        sc = actions[:, 0:2]
        sz = (actions[:, 2] + 1) #(-1, 1) --> (0, 2) 
        sz = sz.view((batch_size, 1))

        
         # TODO: This is the thing that we need to control.
        input = images.squeeze(1)
        output_tensor = self.eyes(input, sc, sz) # (B, 144)
        
        sensor_readings = output_tensor

        # # Process with RNN
        # crops = crops.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, crop_size^2)
        rnn_input = output_tensor.view(batch_size, 1, self.num_kernels)
        
        
        out, hn = self.rnn(rnn_input, h0)
        
        # Predict class and next crop center
        class_pred = self.fc_class(out[:, -1, :])  # Class prediction
        action_pred = torch.tanh(self.fc_action(out[:, -1, :]))  # Action (next crop center)
        

        return class_pred, action_pred, hn, sensor_readings