import torch
import torch.nn as nn
import cv2
import numpy as np

class SingleKernelRetinalGlimpse(nn.Module):
    def __init__(self, input_dim):
        """
        Args:
            input_dim: Tuple (H, W) for the input image dimensions.
        """
        super(SingleKernelRetinalGlimpse, self).__init__()
        self.input_dim = input_dim
        
        self.num_kernels = 144

        self.init_kernel_parameters(sigma_val=.05)

    def init_kernel_parameters(self, sigma_val):
        # Set up the kernels in a grid
        grid_size = int(self.num_kernels ** 0.5)
        linspace = torch.linspace(-1, 1, grid_size)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing="ij")
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        self.mu = nn.Parameter(grid_points)
        self.sigma = nn.Parameter(torch.ones(self.num_kernels)*sigma_val)
                
    def print_picture(self, image):
        """
        Display the image using OpenCV and overlay the kernel centers (mu) as circles.
        The radius of each circle is proportional to the corresponding sigma value.
        The image is upscaled to 512x512 for visualization.
        
        Args:
            image: A (H, W) or (H, W, C) image tensor or numpy array.
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        # Handle grayscale images
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Normalize image to 0-255 for OpenCV if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Upscale the image to 512x512
        H, W = 512, 512
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # Convert mu and sigma from normalized [-1, 1] to pixel coordinates
        pixel_mu = (self.mu + 1) * 0.5 * torch.tensor([W, H]).to(self.mu.device)  # Scale to image dimensions
        pixel_mu = pixel_mu.detach().cpu().numpy()  # Convert to numpy array
        pixel_sigma = (self.sigma * 0.5 * max(W, H)).detach().cpu().numpy()  # Scale sigma to pixel dimensions
        
        # Plot kernel centers on the image
        for mu, sigma in zip(pixel_mu, pixel_sigma):
            center = tuple(int(x) for x in mu)  # Convert to (x, y) tuple
            radius = int(sigma)  # Use sigma as radius
            cv2.circle(image, center, radius, (0, 255, 0), 2)  # Draw circle with green color

        # Display the image
        cv2.imwrite("final_img.png", image)

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
        mu = (s_c.unsqueeze(1) - self.mu) * s_z.unsqueeze(1)  # (B, num_kernels, 2)
        sigma = self.sigma.unsqueeze(0) * s_z  # (B,num_kernels)
        

        # Generate sampling grid
        x = torch.linspace(-1, 1, W, device=device)  # (W,)
        y = torch.linspace(-1, 1, H, device=device)  # (H,)    
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")  # Use indexing='ij' for row-major indexing
        
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W) bc all batches and all kernels
        
        sigma = sigma.view(B, self.num_kernels, 1, 1)

        # Compute Gaussian kernel
        kernels_x = torch.exp(-0.5 * ((grid_x - mu[..., 0].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels_y = torch.exp(-0.5 * ((grid_y - mu[..., 1].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels = kernels_x * kernels_y  # (B, num_kernels, H, W)
        
        kernels /= kernels.sum(dim=(-2, -1), keepdim=True) # Normalize kernel
        
        # Compute weighted sum
        output = (U.unsqueeze(1) * kernels).sum(dim=(-2, -1)) # (B, num_kernels)

        return output

# Example usage
if __name__ == "__main__":
    # Input image (batch size, height, width)
    batch_size = 2
    input_dim = (32, 32)
    U = torch.rand(batch_size, *input_dim)
    U[:, :, :] = 0
    U[:, 14:18, 14:18] = 1

    # Control variables
    s_c = torch.zeros((batch_size, 2))  #torch.rand(batch_size, 2) * 2 - 1  # Positions in [-1, 1]
    s_z = torch.ones((batch_size, 1))     # Zoom factor > 1. NO ZOOM

    # Glimpse model
    model = SingleKernelRetinalGlimpse(input_dim)
    output = model(U, s_c, s_z)

    print("Output:", output)  # Single value per image in the batch
    
    model.print_picture(U[0])