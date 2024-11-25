'''
Authors: Sriram, Varun, and Drew
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
import heapq


class GlimpseModel(nn.Module):
    '''Take in a batch of images, apply (learnable) gaussian kernels, and output batch of sampled tensors'''
    def __init__(self, image_shape, num_kernels=144, device="cuda:0"):
        super(GlimpseModel, self).__init__()
        
        self.image_shape = image_shape
        self.glimpse_window = (50, 50)
        self.device = device
        
        self.num_kernels = num_kernels  # chosen by Cheung et al.
        
        self.init_kernel_parameters()
    
    def init_kernel_parameters(self):
        normalized_length = self.get_normalized_len_from_px_length(self.glimpse_window[0])
        
        starting_sigma_pixels = 2
        starting_sigma_normalize = self.get_normalized_len_from_px_length(starting_sigma_pixels) 
        
        grid_size = int(np.sqrt(self.num_kernels))
        linspace = torch.linspace(-normalized_length/2, normalized_length/2, grid_size)
        grid_x, grid_y = torch.meshgrid(linspace, linspace, indexing="ij")
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        
        mu_init = grid_points # num_kernels by 2
        sigma_init = torch.ones(self.num_kernels, ) * starting_sigma_normalize # learnable num_kernels
        
        # self.mu = nn.Parameter(mu_init.to(self.device))
        self.mu = mu_init.to(self.device)
        # self.sigma = nn.Parameter(sigma_init.to(self.device))
        self.sigma = sigma_init.to(self.device) # Make this learnable
        
    
    # Assumes input_range is (-1, 1)
    def get_normalized_len_from_px_length(self, px_len):
        return px_len * (2/self.image_shape[0])

    def plot_image(self,fname, image, sc, sz, sensor_readings):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            
        k = 20
        k_largest_indices = heapq.nlargest(k, range(len(sensor_readings)), key=lambda x: sensor_readings[x])
        real_k_largest_indices = []
        for ind in k_largest_indices:
            if sensor_readings[ind] > 0.001:
                real_k_largest_indices.append(ind)
                
        k_largest_indices = real_k_largest_indices
                
            
        print("k_largest_indices", k_largest_indices)
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Normalize image to 0-255 for OpenCV if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Upscale the image to 512x512
        H, W = 512, 512
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_NEAREST)
        
        shifted_mu = (sc + self.mu) * sz # (num_kernels, 2)
        shifted_sigma = (self.sigma * sz).squeeze(0) # (num_kernels,)
            
        # Convert mu and sigma from normalized [-1, 1] to pixel coordinates
        pixel_mu = (shifted_mu + 1) * 0.5 * torch.tensor([W, H]).to(shifted_mu.device)  # Scale to image dimensions
        pixel_mu = pixel_mu.detach().cpu().numpy()  # Convert to numpy array
        pixel_sigma = (shifted_sigma * 0.5 * max(W, H)).detach().cpu().numpy()  # Scale sigma to pixel dimensions
        
        # Plot kernel centers on the image
        idx = 0
        for mu, sigma in zip(pixel_mu, pixel_sigma):
            # print(idx, "mu = ", mu)
            center = tuple(int(x) for x in mu)  # Convert to (x, y) tuple
            radius = abs(int(sigma.item()))  # Use sigma as radius
            
            color = (255, 255, 0)
            if idx in k_largest_indices:
                color = (255, 0, 0)
            cv2.circle(image, center, radius, color, thickness=1)
                        
            idx +=1

        cv2.imwrite(fname, image)
        
    # s_c: centre of the glimple
    # s_z: zoom, we're gonna keep it 1 for now
    def forward(self, imgs, s_c, s_z):
        B, H, W = imgs.shape
        device = imgs.device
        
        # Compute Kernel Center and Sigma (Eqn 2/3 from the paper)
        mu = (s_c.unsqueeze(1) + self.mu)*s_z.unsqueeze(1) # (B, num_kernels, 2) 
        sigma = self.sigma * s_z # (B, num_kernels)        
        # Generate sampling grid
        x = torch.linspace(-1, 1, W, device=device)  # (W,)
        y = torch.linspace(-1, 1, H, device=device)  # (H,)    
        grid_y, grid_x = torch.meshgrid(x, y, indexing="ij")
        
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        sigma = sigma.view(B, self.num_kernels, 1, 1)   
        
        # Compute the gaussian kernel
        kernels_x = torch.exp(-0.5 * ((grid_x- mu[..., 0].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels_y = torch.exp(-0.5 * ((grid_y - mu[..., 1].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
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
    U_old = read_image("./dataset/cluttered_mnist/3/11.png").unsqueeze(0)
    image_shape = (100, 100)
    model = GlimpseModel(image_shape).to("cuda:0")
    
    U = torch.zeros((1, 100, 100), device="cuda:0")
    # U[:, :, :] = U_old.to("cuda:0")
    # U[:, 40:60, 40:60] = 1
    
    U[:, 90:100, 0:10] = 1
    U[:, 15:35, 40:80] = 1
    U[:, 0:5, 50:60] = 1
    
    batch_size = len(U)
    
    
    s_c = torch.rand((batch_size, 2), device="cuda:0") * 2 -1
    s_z = torch.ones((batch_size, 1), device="cuda:0")
    
    # s_c[0, 0] = -0.5
    # s_c[0, 1] = 0.5
            
    sensor_reading = model(U, s_c, s_z)
    # print("This is the output", sensor_reading)
    print("Plotting now::")
    for i in range(batch_size):
        model.plot_image("test{}.png".format(i), U[i], s_c[i], s_z[i], sensor_reading[0])
    
    print("Plotting Finished!")


if __name__ == '__main__':
    main()
