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
    # Vérifier les NaN dans les entrées
    if torch.isnan(pred).any() or torch.isnan(target).any():
        return torch.tensor(1.0, device=pred.device, requires_grad=True)
    
    B, C, H, W = pred.shape
    
    # Vérifier les dimensions minimales pour éviter les erreurs de convolution
    if H < window_size or W < window_size:
        # Fallback: utiliser MSE si les dimensions sont trop petites
        mse = F.mse_loss(pred, target)
        return mse / (data_range ** 2)  # Normaliser par rapport au data_range
    
    kernel = gaussian_kernel(window_size, sigma, channels=C).to(pred.device)
    
    try:
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

        # Éviter la division par zéro
        numerator = (2*mu1_mu2 + C1)*(2*sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + eps
        
        ssim_map = numerator / denominator
        
        # Clamper SSIM pour éviter des valeurs extrêmes
        ssim_map = torch.clamp(ssim_map, min=0.0, max=1.0)
        
        ssim_mean = ssim_map.mean([1,2,3])
        loss = 1 - ssim_mean.mean()
        
        # Vérifier si la loss résultante est valide
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1.0, device=pred.device, requires_grad=True)
        
        return loss
        
    except Exception as e:
        # En cas d'erreur, fallback vers MSE
        print(f"SSIM computation failed: {e}, falling back to MSE")
        mse = F.mse_loss(pred, target)
        return mse / (data_range ** 2)
