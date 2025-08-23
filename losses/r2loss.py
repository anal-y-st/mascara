import torch
import torch.nn as nn

class R2Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, y_pred, y_true):
        # Vérifier les NaN dans les entrées
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            return torch.tensor(float('inf'), device=y_pred.device, requires_grad=True)
        
        y_true_mean = torch.mean(y_true, dim=[1,2,3], keepdim=True)
        ss_tot = torch.sum((y_true - y_true_mean)**2, dim=[1,2,3]) + self.eps
        ss_res = torch.sum((y_true - y_pred)**2, dim=[1,2,3])
        
        # Éviter la division par zéro et les valeurs négatives
        ss_tot = torch.clamp(ss_tot, min=self.eps)
        r2 = 1 - ss_res / ss_tot
        
        # Clamper R2 pour éviter des valeurs extrêmes
        r2 = torch.clamp(r2, min=-10.0, max=1.0)
        
        # Retourner la loss (1 - R2) avec gestion des NaN
        loss = 1 - r2.mean()
        
        # Vérifier si la loss résultante est valide
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(1.0, device=y_pred.device, requires_grad=True)
        
        return loss
