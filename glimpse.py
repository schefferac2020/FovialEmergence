'''
Authors: Sriram, Varun, and Drew
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
import heapq
import torch.nn.functional as F
from enum import Enum

## Some util functions
def plot_glimpse_image(fname, image, mu, sigma, sc, sz, sensor_readings):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        
    k = 20
    k_largest_indices = heapq.nlargest(k, range(len(sensor_readings)), key=lambda x: sensor_readings[x])
    real_k_largest_indices = []
    for ind in k_largest_indices:
        if sensor_readings[ind] > 0.001:
            real_k_largest_indices.append(ind)
            
    k_largest_indices = real_k_largest_indices
        
    # print("k_largest_indices", k_largest_indices)
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Normalize image to 0-255 for OpenCV if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Upscale the image to 512x512
    H, W = (512, 512)
    upscale_factor= H/image.shape[0]
    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
    
    shifted_mu = (sc + mu) * sz # (num_kernels, 2)
    shifted_sigma = (sigma * sz).squeeze(0) # (num_kernels,)
        
    # Convert mu and sigma from normalized [-1, 1] to pixel coordinates
    pixel_mu = (shifted_mu + 1) * 0.5 * torch.tensor([W, H]).to(shifted_mu.device)  # Scale to image dimensions
    pixel_mu = pixel_mu.detach().cpu().numpy()  # Convert to numpy array
    pixel_sigma = (shifted_sigma*upscale_factor).detach().cpu().numpy()  # Scale sigma to pixel dimensions
    
    # Plot kernel centers on the image
    idx = 0
    for mu, sigma in zip(pixel_mu, pixel_sigma):
        # print(idx, "mu = ", mu)
        center = tuple(int(x) for x in mu)  # Convert to (x, y) tuple
        radius = abs(int(sigma.item()))  # Use sigma as radius
        
        color = (255, 255, 0)
        if idx in k_largest_indices:
            color = (0, 0, 255)
        cv2.circle(image, center, radius, color, thickness=1)
                    
        idx +=1

    cv2.imwrite(fname, image)
    
# Assumes input_range is (-1, 1)
def get_normalized_len_from_px_length(px_len, image_shape=(100, 100)):
    return px_len * (2/image_shape[0])

def get_px_from_normalized(normalized_tensor, image_shape=(100, 100)):
    '''convert [-1, 1] --> [0, H]'''
    return ((normalized_tensor + 1)/2 * (image_shape[0]-1)).round().long()



class GlimpseModelFaster(nn.Module):
    '''
    Take in a batch of images, apply (learnable) gaussian kernels, and output batch of sampled tensors
    NOTE: This is a faster implimentation using convolutions of FIXED SIZE kernels (i.e. 11x11). Thus, in
    its current state it does not support scaling actions (sz)
    '''
    def __init__(self, image_shape, num_kernels=144, kernel_max_rf=11, init_glimpse_window=(50, 50), init_sigma=2, device="cpu"):
        super(GlimpseModelFaster, self).__init__()
        
        self.image_shape = image_shape
        self.num_kernels = num_kernels
        self.kernel_len = kernel_max_rf
        self.device = device
        
        self.init_kernel_parameters(init_glimpse_window, init_sigma)
    
    def init_kernel_parameters(self, glimpse_window, init_sigma_px):
        # 1. Initialize the Grid of Kernel Positions, mu (normalized units from -1 to 1)
        normalized_length = get_normalized_len_from_px_length(glimpse_window[0], self.image_shape)
        grid_size = int(np.sqrt(self.num_kernels))
        linspace = torch.linspace(-normalized_length/2, normalized_length/2, grid_size)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing="ij")
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        mu_init = grid_points # (num_kernels, 2)
        
        # 2. Initialize the sigmas for each kernel (pixel units)
        sigma_init = torch.ones(self.num_kernels, ) * init_sigma_px #  (num_kernels, )
        
        # Make mu and sigma learnable 
        self.mu = nn.Parameter(mu_init.to(self.device))
        self.sigma = nn.Parameter(sigma_init.to(self.device))
        
        
    def get_2D_gaussian_kernel(self, len_pixels, sigma_pixels):
        kernel_1d = torch.signal.windows.gaussian(len_pixels, std=sigma_pixels)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def create_gaussian_kernels(self, kernel_len, sigma_px):
        """
        Precompute Gaussian kernels of size (kernel_size x kernel_size) for each kernel.
        """
        kernels = []
                
        for i in range(self.num_kernels):
            kernel = self.get_2D_gaussian_kernel(kernel_len, sigma_px[i])
            kernels.append(kernel.unsqueeze(0))

        return torch.cat(kernels, dim=0).to(self.device)
        
    
    def forward(self, imgs, s_c, s_z):
        '''
        imgs: (B, H, W) batch of images
        s_c: (B, 2) center shifts for each image in the batch (for x and y coordinates)
        s_z: (B,) zoom factor for scaling sigma
        '''
        
        # TODO: I think this might only work for fixed zoom at the moment? 
        
        B, _, _ = imgs.shape
        
        # Adjust kernel centers based on s_c and s_z (zoom factor)
        mu_normalized = (s_c.unsqueeze(1) + self.mu)  # (B, num_kernels, 2) 
        mu_pixel = get_px_from_normalized(mu_normalized, self.image_shape)
        mu_pixel_x = mu_pixel[:, :, 0]
        mu_pixel_y = mu_pixel[:, :, 1]
        
        # Adjust sigma based on zoom
        sigma_normalized = self.sigma  # (num_kernels)
        sigma_px = get_px_from_normalized(sigma_normalized, self.image_shape)
        
        
        self.kernels = self.create_gaussian_kernels(self.kernel_len, sigma_px) #TODO: Maybe take in the mu values? How off in terms of pixels right?
        
        imgs = imgs.unsqueeze(1)
        self.kernels = self.kernels.unsqueeze(1)
        output = F.conv2d(imgs,   
                       self.kernels,
                       stride=1,
                       padding=(self.kernel_len-1)//2)
        
        B, num_ker, H, W = output.shape
        
        batch_indices = torch.arange(B, device=output.device).unsqueeze(1)  # (B, 1)
        kernel_indices = torch.arange(num_ker, device=output.device).unsqueeze(0)  # (1, num_ker)
        res = output[batch_indices, kernel_indices, mu_pixel_y.clamp(0, H - 1), mu_pixel_x.clamp(0, W - 1)] 
        
        valid_x = (mu_pixel_x >= 0) & (mu_pixel_x < W)
        valid_y = (mu_pixel_y >= 0) & (mu_pixel_y < H)
        valid_mask = valid_x & valid_y
        
        return res*valid_mask


class GlimpseModel(nn.Module):
    '''Take in a batch of images, apply (learnable) gaussian kernels, and output batch of sampled tensors'''
    def __init__(self, image_shape, num_kernels=144, init_glimpse_window=(50, 50), init_sigma=2, device="cpu"):
        super(GlimpseModel, self).__init__()
        
        self.image_shape = image_shape
        self.glimpse_window = (50, 50)
        self.device = device
        
        self.num_kernels = num_kernels  # chosen by Cheung et al.
        
        self.init_kernel_parameters(init_glimpse_window, init_sigma)
        
        x = torch.linspace(-1, 1, image_shape[0], device=self.device)  # (W,)
        y = torch.linspace(-1, 1, image_shape[1], device=self.device)  # (H,)    
        grid_y, grid_x = torch.meshgrid(x, y, indexing="ij")
        self.grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def init_kernel_parameters(self, glimpse_window, init_sigma_px):
        # 1. Initialize the Grid of Kernel Positions, mu (normalized units from -1 to 1)
        normalized_length = get_normalized_len_from_px_length(glimpse_window[0], self.image_shape)
        grid_size = int(np.sqrt(self.num_kernels))
        linspace = torch.linspace(-normalized_length/2, normalized_length/2, grid_size)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing="ij")
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        mu_init = grid_points # (num_kernels, 2)
        
        # 2. Initialize the sigmas for each kernel (pixel units)
        sigma_init = torch.ones(self.num_kernels, ) * init_sigma_px #  (num_kernels, )
        
        # Make mu and sigma learnable 
        self.mu = nn.Parameter(mu_init.to(self.device))
        self.sigma = nn.Parameter(sigma_init.to(self.device))
        
    def forward(self, imgs, s_c, s_z):
        '''
        imgs: (B, 100, 100)
        '''
        
        B, H, W = imgs.shape
        device = imgs.device
        
        # Compute Kernel Center and Sigma (Eqn 2/3 from the paper)
        mu = (s_c.unsqueeze(1) + self.mu)*s_z.unsqueeze(1) # (B, num_kernels, 2) 
        sigma = get_normalized_len_from_px_length(self.sigma) * s_z # (B, num_kernels)        

        sigma = sigma.view(B, self.num_kernels, 1, 1)           
        
        # Compute the gaussian kernel
        kernels_x = torch.exp(-0.5 * ((self.grid_x- mu[..., 0].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels_y = torch.exp(-0.5 * ((self.grid_y - mu[..., 1].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels = kernels_x*kernels_y   # (B, num_kernels, H, W)
        
        # normalize
        kernels /= (kernels.sum(dim=(-2, -1), keepdim=True) + 1e-7)        
        
        # Compute the weighted sum
        output = (imgs.unsqueeze(1) * kernels).sum(dim=(-2, -1)) # (B, num_kernels)
        
        return output

def read_image(file_pth):
    image = cv2.imread(file_pth, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Can't open {file_pth}")
        exit(0)
    image = image / 255.0
    return torch.tensor(image, dtype=torch.float32)

def main():
    image_shape = (100, 100)
    device = "cpu"
    
    batch_size = 16
    images = torch.zeros((batch_size, 100, 100), device=device)
    # U[:, :, :] = read_image("./dataset/cluttered_mnist/3/11.png").unsqueeze(0).to("cpu")
    images[:, 90:100, 0:10] = 1
    images[:, 15:35, 40:80] = 1
    images[:, 0:5, 50:60] = 1
    
    # Random Commands
    s_c = torch.rand((batch_size, 2), device=device) * 2 -1
    s_z = torch.ones((batch_size, 1), device=device)
    
    # Test slow model
    slow_model = GlimpseModel(image_shape).to(device)
    slow_model_sensor_readings = slow_model(images, s_c, s_z)
    print("Slow model sensor reading:", slow_model_sensor_readings)
    
    # Test fast model 
    fast_model = GlimpseModelFaster(image_shape).to(device)
    fast_model_sensor_readings = fast_model(images, s_c, s_z)
    print("Fast model sensor reading:", fast_model_sensor_readings)
    
    print("Plotting now:")
    plot_glimpse_image("fast_glimpse.png", images[0], fast_model.mu, fast_model.sigma, s_c[0], s_z[0], fast_model_sensor_readings[0])
    plot_glimpse_image("slow_glimpse.png", images[0], slow_model.mu, slow_model.sigma, s_c[0], s_z[0], slow_model_sensor_readings[0])


if __name__ == '__main__':
    main()
