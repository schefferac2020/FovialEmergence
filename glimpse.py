import torch
import torch.nn as nn

class SingleKernelRetinalGlimpse(nn.Module):
    def __init__(self, input_dim):
        """
        Args:
            input_dim: Tuple (H, W) for the input image dimensions.
        """
        super(SingleKernelRetinalGlimpse, self).__init__()
        self.input_dim = input_dim

        # Learnable kernel parameters
        self.mu = nn.Parameter(torch.tensor([0.0, 0.0]))  # Center of the kernel
        self.sigma = nn.Parameter(torch.tensor(0.1))      # Spread (standard deviation)

    def forward(self, U, s_c, s_z):
        """
        Args:
            U: Input image tensor of shape (B, H, W).
            s_c: Control position (B, 2), normalized coordinates (-1, 1).
            s_z: Control zoom (B, 1), a positive scalar zoom factor.
        Returns:
            output: A single value per image in the batch (B,).
        """
        B, H, W = U.size()
        device = U.device

        # Compute kernel center and spread
        mu = (s_c - self.mu) * s_z  # (B, 2)
        sigma = self.sigma * s_z  # (B,)

        # Generate sampling grid
        x = torch.linspace(-1, 1, W, device=device)  # (W,)
        y = torch.linspace(-1, 1, H, device=device)  # (H,)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")  # Use indexing='ij' for row-major indexing
        print("This is grid_x: ", grid_x)
        print("This is grid_y: ", grid_y)
        
        print("subtraction:", grid_x - mu[:, 0])


        # Compute Gaussian kernel
        kernel_x = torch.exp(-0.5 * ((grid_x - mu[:, 0].view(-1, 1, 1)) ** 2) / (sigma ** 2).view(-1, 1, 1))
        kernel_y = torch.exp(-0.5 * ((grid_y - mu[:, 1].view(-1, 1, 1)) ** 2) / (sigma ** 2).view(-1, 1, 1))
        print("kernel_x.shape", kernel_x.shape)
        print("kernel_y.shape", kernel_y.shape)
        
        kernel = kernel_x * kernel_y  # (B, H, W)
        
        
        kernel /= kernel.sum(dim=(-2, -1), keepdim=True) # Normalize kernel
        
        print("This is the kernel", kernel.shape)
        print(kernel[0, 12:20, 12:20])

        # Compute weighted sum
        output = (U * kernel).sum(dim=(-2, -1))  # (B,)

        return output

# Example usage
if __name__ == "__main__":
    # Input image (batch size, height, width)
    batch_size = 1
    input_dim = (32, 32)
    U = torch.rand(batch_size, *input_dim)
    U[:, :, :] = 0
    U[:, 16, 16] = 1

    # Control variables
    s_c = torch.rand(batch_size, 2) * 0 #torch.rand(batch_size, 2) * 2 - 1  # Positions in [-1, 1]
    s_z = 1      # Zoom factor > 1. NO ZOOM

    # Glimpse model
    model = SingleKernelRetinalGlimpse(input_dim)
    output = model(U, s_c, s_z)

    print("Output:", output)  # Single value per image in the batch
