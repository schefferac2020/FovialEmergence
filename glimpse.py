'''
Authors: Sriram, Varun, and Drew
'''

import torch
import torch.nn as nn
import cv2
import numpy as np
import heapq
import torch.nn.functional as F


class GlimpseModel(nn.Module):
    '''Take in a batch of images, apply (learnable) gaussian kernels, and output batch of sampled tensors'''
    def __init__(self, image_shape, num_kernels=144, device="cpu"):
        super(GlimpseModel, self).__init__()
        
        self.image_shape = image_shape
        self.glimpse_window = (50, 50)
        self.device = device
        
        self.num_kernels = num_kernels  # chosen by Cheung et al.
        
        self.init_kernel_parameters()
        
        
        x = torch.linspace(-1, 1, image_shape[0], device=self.device)  # (W,)
        y = torch.linspace(-1, 1, image_shape[1], device=self.device)  # (H,)    
        grid_y, grid_x = torch.meshgrid(x, y, indexing="ij")
        
        self.grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        self.grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
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
        
        self.mu = nn.Parameter(mu_init.to(self.device))
        # self.mu = mu_init.to(self.device)
        self.sigma = nn.Parameter(sigma_init.to(self.device))
        # self.sigma = sigma_init.to(self.device) # Make this learnable
        
        self.kernel_len = 11
        self.kernels = self._create_gaussian_kernels(self.kernel_len)
        
        
    def get_2D_gaussian_kernel(self, len_pixels, sigma_pixels):
        kernel_1d = torch.signal.windows.gaussian(len_pixels, std=sigma_pixels)
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        return kernel_2d

    def _create_gaussian_kernels(self, kernel_len):
        """
        Precompute Gaussian kernels of size (kernel_size x kernel_size) for each kernel.
        """
        kernels = []
        
        sigma = 1 #(pixel)
        
        for i in range(self.num_kernels):
            kernel = self.get_2D_gaussian_kernel(kernel_len, sigma)
            kernels.append(kernel.unsqueeze(0))

        return torch.cat(kernels, dim=0).to(self.device)
        
    
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
                
            
        # print("k_largest_indices", k_largest_indices)
        
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
                color = (0, 0, 255)
            cv2.circle(image, center, radius, color, thickness=1)
                        
            idx +=1

        cv2.imwrite(fname, image)
        
    # s_c: centre of the glimple
    # s_z: zoom, we're gonna keep it 1 for now
    def forward(self, imgs, s_c, s_z):
        '''
        imgs: (B, 100, 100)
        '''
        
        B, H, W = imgs.shape
        device = imgs.device
        print(1)
        
        # Compute Kernel Center and Sigma (Eqn 2/3 from the paper)
        mu = (s_c.unsqueeze(1) + self.mu)*s_z.unsqueeze(1) # (B, num_kernels, 2) 
        sigma = self.sigma * s_z # (B, num_kernels)        
        # Generate sampling grid
        print(2)

        sigma = sigma.view(B, self.num_kernels, 1, 1)           
        
        # Compute the gaussian kernel
        kernels_x = torch.exp(-0.5 * ((self.grid_x- mu[..., 0].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        kernels_y = torch.exp(-0.5 * ((self.grid_y - mu[..., 1].view(B, self.num_kernels, 1, 1)) ** 2) / sigma ** 2)
        print(3)
        
        
        kernels = kernels_x*kernels_y   # (B, num_kernels, H, W)
        print(4)
        

        # normalize
        kernels /= (kernels.sum(dim=(-2, -1), keepdim=True) + 1e-7)
        print(5)
        
        
        # Compute the weighted sum
        output = (imgs.unsqueeze(1) * kernels).sum(dim=(-2, -1)) # (B, num_kernels)
        print(6)
        
        
        return output
    
    def forward_2(self, imgs, s_c, s_z):
        '''
        imgs: (B, H, W) batch of images
        s_c: (B, 2) center shifts for each image in the batch (for x and y coordinates)
        s_z: (B,) zoom factor for scaling sigma
        '''
        B, H, W = imgs.shape
        
        # Adjust kernel centers based on s_c and s_z (zoom factor)
        mu_normalized = (s_c.unsqueeze(1) + self.mu) * s_z.unsqueeze(1)  # (B, num_kernels, 2) 
        mu_pixel = ((mu_normalized + 1) * 100/2).round().long()
        mu_pixel_x = mu_pixel[:, :, 0].clamp(0, 100 - 1)
        mu_pixel_y = mu_pixel[:, :, 1].clamp(0, 100 - 1)
        
        # Adjust sigma based on zoom
        sigma = self.sigma * s_z.view(-1, 1)  # (B, num_kernels)
        
        
        self._create_gaussian_kernels(10) #TODO: Maybe take in the mu values? How off in terms of pixels right?
        
        imgs = imgs.unsqueeze(1)
        self.kernels = self.kernels.unsqueeze(1)
        output = F.conv2d(imgs.transpose(),   
                       self.kernels,
                       stride=1,
                       padding=(self.kernel_len-1)//2)
        
        B, num_ker, W, H = output.shape
        
        res = torch.zeros(B, num_ker)
        for b in range(B):
            for ker in range(num_ker):
                image = output[b, ker]
                val = image[mu_pixel[b, ker, 0].clamp(0, 100 - 1), mu_pixel[b, ker, 1].clamp(0, 100 - 1)]
                
                res[b, ker] = val
        
        return res

        print("This is the shape", output.shape)

        # Vectorized indexing to extract values at (mu_x, mu_y)
        res = output[torch.arange(B).unsqueeze(1), torch.arange(num_ker), mu_pixel_x, mu_pixel_y]
        
        return res
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
    model = GlimpseModel(image_shape).to("cpu")
    
    U = torch.zeros((15, 100, 100), device="cpu")
    # U[:, :, :] = U_old.to("cpu")
    # U[:, 60:80, 60:80] = 1
    # U[:, 60:80, 60:80] = 1
    
    U[:, 90:100, 0:10] = 1
    U[:, 15:35, 40:80] = 1
    U[:, 0:5, 50:60] = 1
    
    batch_size = len(U)
    
    
    s_c = torch.rand((batch_size, 2), device="cpu") * 2 -1
    s_z = torch.ones((batch_size, 1), device="cpu")
    
    # s_c[0, 0] = -0.5
    # s_c[0, 1] = 0.5
            
    # sensor_reading = model(U, s_c, s_z)
    # print("Finished first forward model")
    sensor_reading = model.forward_2(U, s_c, s_z)
    print(sensor_reading)
    print("Finished second forward model")
    
    # print("This is the output", sensor_reading)
    print("Plotting now::")
    for i in range(batch_size):
        model.plot_image("test{}.png".format(i), U[i], s_c[i], s_z[i], sensor_reading[i])
    
    print("Plotting Finished!")


if __name__ == '__main__':
    main()
