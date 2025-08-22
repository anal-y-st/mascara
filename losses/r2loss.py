import torch
import torch.nn as nn

class R2Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true, dim=[1,2,3], keepdim=True)
        ss_tot = torch.sum((y_true - y_true_mean)**2, dim=[1,2,3]) + self.eps
        ss_res = torch.sum((y_true - y_pred)**2, dim=[1,2,3])
        r2 = 1 - ss_res / ss_tot
        return torch.mean(1 - r2)
