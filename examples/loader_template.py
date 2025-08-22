import torch
from torch.utils.data import DataLoader, TensorDataset

def get_loaders(config: dict):
    """Exemple minimal purement synthétique.
    Remplacez par vos propres Dataset/DataLoader.
    """
    B, C, H, W = 32, config["model"]["in_ch"], config["model"]["img_size"][0], config["model"]["img_size"][1]
    X_train = torch.randn(B, C, H, W)
    y_train = torch.randn(B, 1, H, W)  # cible pixel-wise

    X_val = torch.randn(B, C, H, W)
    y_val = torch.randn(B, 1, H, W)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

    stats = dict(input_mean=0.0, input_std=1.0, label_mean=0.0, label_std=1.0)
    return train_loader, val_loader, stats
