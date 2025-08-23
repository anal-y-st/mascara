import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_numpy(y_true, y_pred):
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return dict(r2=r2, mae=mae, rmse=rmse)

def evaluate_model(model, loader, device, label_mean=None, label_std=None):
    model.eval()
    y_trues, y_preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            # rescale to original label units if normalized
            if (label_mean is not None) and (label_std is not None):
                pred_res = (pred.cpu().numpy() * label_std) + label_mean
                yb_res = (yb.cpu().numpy() * label_std) + label_mean
            else:
                pred_res = pred.cpu().numpy()
                yb_res = yb.cpu().numpy()
            y_preds.append(pred_res.ravel())
            y_trues.append(yb_res.ravel())
    y_pred = np.concatenate(y_preds)
    y_true = np.concatenate(y_trues)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse
