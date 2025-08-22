import torch
import torch.nn.functional as F

def gaussian_kernel(window_size=11, sigma=1.5, channels=1):
    coords = torch.arange(window_size).float() - (window_size-1)/2.0
    g = torch.exp(-(coords**2)/(2*sigma*sigma))
    g = g / g.sum()
    kernel = g[:,None] * g[None,:]
    kernel = kernel.expand(channels, 1, window_size, window_size)
    return kernel

def ssim_loss(pred, target, window_size=11, sigma=1.5, data_range=1.0, eps=1e-6):
    B, C, H, W = pred.shape
    kernel = gaussian_kernel(window_size, sigma, channels=C).to(pred.device)
    mu1 = F.conv2d(pred, kernel, padding=window_size//2, groups=C)
    mu2 = F.conv2d(target, kernel, padding=window_size//2, groups=C)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred*pred, kernel, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target*target, kernel, padding=window_size//2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred*target, kernel, padding=window_size//2, groups=C) - mu1_mu2

    C1 = (0.01*data_range)**2
    C2 = (0.03*data_range)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + eps)
    ssim_mean = ssim_map.mean([1,2,3])
    return torch.mean(1 - ssim_mean)
